"""Run the AthenaK fixture and compare every RK stage in float64."""

import argparse
import csv
from pathlib import Path
import subprocess

import jax
import jax.numpy as jnp
import numpy as np

from jathenax.fluxes import RECONSTRUCTIONS, RIEMANN_SOLVERS
from jathenax.hydro import RK_STAGES, hydro_step


def read_stage(path, fields):
    with path.open() as file:
        header = tuple(map(float, file.readline().split()))
        ny, nx = map(int, header[:2])
        shape = (int(header[8]), ny, nx) if len(header) > 8 else (ny, nx)
        nvar = 5 if len(shape) == 3 else 3
        data = np.loadtxt(file).reshape(*shape, len(fields), nvar)
    return header, {name: data[..., i, :] for i, name in enumerate(fields)}


def energy_flux_scale(W, normal, gamma):
    rho, v, e = W[..., 0], W[..., 1:4], W[..., 4]
    sound = np.sqrt(gamma * (gamma - 1.0) * e / rho)
    return (abs(v[..., normal]) + sound) * (gamma * e + 0.5 * rho * np.sum(v**2, axis=-1))


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("athena", type=Path, help="AthenaK built with the stage fixture")
parser.add_argument("output", type=Path, help="Directory for reference and JAX outputs")
parser.add_argument("--viscosity", type=float, default=0.0)
parser.add_argument("--smagorinsky-cs", type=float, default=0.0)
parser.add_argument("--reconstruct", choices=RECONSTRUCTIONS, default="ppm4")
parser.add_argument("--rsolver", choices=RIEMANN_SOLVERS, default="roe")
parser.add_argument("--ideal-3d", action="store_true", help="Check all five conserved fields in 3D")
args = parser.parse_args()
if args.rsolver == "hllc" and not args.ideal_3d:
    parser.error("the AthenaK HLLC reference requires --ideal-3d")
directions = "xyz" if args.ideal_3d else "xy"
fields = ("primitives", *(f"{side}_{d}" for d in directions for side in ("left", "right")),
          *(f"flux_{d}" for d in directions), "saved", "conserved")
overrides = ["mesh/nx3=8", "meshblock/nx3=8", "hydro/eos=ideal", "hydro/gamma=1.6666666666666667",
             "hydro/pfloor=1e-10"] if args.ideal_3d else []
athena = args.athena.resolve()
deck = Path(__file__).with_name("athenak_stages.athinput").resolve()
jax.config.update("jax_enable_x64", True)

rows = []
cases = [("smooth", 0.0), ("jump", 2.5)]
if args.rsolver == "hllc":
    cases.append(("moderate_jump", 0.5))
for case, jump_speed in cases:
    for integrator, coefficients in RK_STAGES.items():
        folder = args.output / case / integrator
        folder.mkdir(parents=True)
        with (folder / "athenak.log").open("w") as log:
            subprocess.run(
                [str(athena), "-i", str(deck), f"time/integrator={integrator}",
                 f"problem/velocity_jump={'true' if jump_speed else 'false'}",
                 f"problem/jump_speed={jump_speed}",
                 f"hydro/viscosity={args.viscosity}",
                 f"hydro/smagorinsky_cs={args.smagorinsky_cs}",
                 f"hydro/reconstruct={args.reconstruct}", f"hydro/rsolver={args.rsolver}", *overrides],
                cwd=folder, stdout=log, stderr=subprocess.STDOUT, check=True,
            )

        reference = [read_stage(path, fields) for path in sorted(folder.glob("stage*.txt"))]
        assert len(reference) == len(coefficients), folder
        header, first = reference[0]
        _, _, dx, dy, cs, dfloor, dt, _ = header[:8]
        spacing = (dx, dy, header[9]) if args.ideal_3d else (dx, dy)
        eos = (dict(eos="ideal", gamma=header[10], pfloor=header[11]) if args.ideal_3d
               else dict(eos="isothermal", cs=cs))
        initial = first["saved"]
        assert dt == 0.001, f"AthenaK took dt={dt}"

        # Both solvers use exactly the reference's initial conserved state and dt.
        step = jax.jit(
            lambda U, dt: hydro_step(
                U, dt, spacing, **eos, dfloor=dfloor, integrator=integrator, trace=True,
                viscosity=args.viscosity, smagorinsky_cs=args.smagorinsky_cs,
                reconstruct=args.reconstruct, rsolver=args.rsolver,
            )
        )
        final, stages = jax.device_get(step(jnp.asarray(initial), dt))
        np.savez(
            folder / "jax_stages.npz", dt=dt, initial=initial, final=final,
            **{f"stage{number}_{name}": value
               for number, stage in enumerate(stages, 1) for name, value in stage.items()},
        )
        np.testing.assert_allclose(
            final.mean(axis=tuple(range(len(spacing)))), initial.mean(axis=tuple(range(len(spacing)))),
            rtol=0, atol=3e-14,
        )

        worst = dict.fromkeys(("face states", "face fluxes", "conserved"), 0.0)
        for number, (ref, computed) in enumerate(zip(reference, stages, strict=True), 1):
            metadata, expected = ref
            np.testing.assert_equal(metadata[:7] + metadata[8:], header[:7] + header[8:])
            np.testing.assert_equal(metadata[7], coefficients[number - 1][2] * dt)
            for field in fields:
                actual = computed[field]
                assert actual.dtype == np.float64
                error = float(np.max(np.abs(actual - expected[field])))
                different = int(np.count_nonzero(
                    actual.view(np.uint64) != expected[field].view(np.uint64)
                ))
                rows.append(dict(
                    case=case, integrator=integrator, stage=number, field=field,
                    max_abs_error=error, differing_values=different, total_values=actual.size,
                ))
                label = (
                    "face states" if field.startswith(("left_", "right_"))
                    else "face fluxes" if field.startswith("flux_") else field
                )
                if label in worst:
                    worst[label] = max(worst[label], error)
                # Opposing flux contributions can cancel at supersonic jumps.
                atol = 4e-14 if field.startswith("flux_") else 2e-14
                reference_field = expected[field]
                if args.ideal_3d and field.startswith("flux_"):
                    # Opposing streams can cancel the energy flux while each carries large enthalpy.
                    d = field[-1]
                    scale = np.maximum(*(energy_flux_scale(expected[f"{side}_{d}"], "xyz".index(d),
                                                          eos["gamma"]) for side in ("left", "right")))
                    bound = atol + 2e-14*abs(reference_field[..., 4]) + 32*np.finfo(float).eps*scale
                    np.testing.assert_array_less(abs(actual[..., 4] - reference_field[..., 4]), bound)
                    actual, reference_field = actual[..., :4], reference_field[..., :4]
                np.testing.assert_allclose(
                    actual, reference_field, rtol=2e-14, atol=atol,
                    err_msg=f"{case} {integrator} stage {number} {field}",
                )
        print(f"{case:6s} {integrator}  {len(stages)} stages  "
              + "  ".join(f"{name} {error:.3e}" for name, error in worst.items()))

with (args.output / "stage_errors.csv").open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
