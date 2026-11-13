"""Run the AthenaK fixture and compare every RK stage in float64."""

import argparse
import csv
from pathlib import Path
import subprocess

import jax
import jax.numpy as jnp
import numpy as np

from jathenax.fluxes import RECONSTRUCTIONS, RIEMANN_SOLVERS
from jathenax.hydro import RK_STAGES, isothermal_step


# These fields follow the columns written by the AthenaK source callback.
fields = (
    "primitives", "left_x", "right_x", "left_y", "right_y",
    "flux_x", "flux_y", "saved", "conserved",
)


def read_stage(path):
    with path.open() as file:
        ny, nx, dx, dy, cs, dfloor, dt, stage_dt = map(float, file.readline().split())
        data = np.loadtxt(file).reshape(int(ny), int(nx), len(fields), 3)
    arrays = {name: data[:, :, index, :] for index, name in enumerate(fields)}
    return (dx, dy, cs, dfloor, dt, stage_dt), arrays


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("athena", type=Path, help="AthenaK built with the stage fixture")
parser.add_argument("output", type=Path, help="Directory for reference and JAX outputs")
parser.add_argument("--viscosity", type=float, default=0.0)
parser.add_argument("--smagorinsky-cs", type=float, default=0.0)
parser.add_argument("--reconstruct", choices=RECONSTRUCTIONS, default="ppm4")
parser.add_argument("--rsolver", choices=RIEMANN_SOLVERS, default="roe")
args = parser.parse_args()
athena = args.athena.resolve()
deck = Path(__file__).with_name("athenak_stages.athinput").resolve()
jax.config.update("jax_enable_x64", True)

rows = []
for case, jump in (("smooth", "false"), ("jump", "true")):
    for integrator, coefficients in RK_STAGES.items():
        folder = args.output / case / integrator
        folder.mkdir(parents=True)
        with (folder / "athenak.log").open("w") as log:
            subprocess.run(
                [str(athena), "-i", str(deck), f"time/integrator={integrator}",
                 f"problem/velocity_jump={jump}", f"hydro/viscosity={args.viscosity}",
                 f"hydro/smagorinsky_cs={args.smagorinsky_cs}",
                 f"hydro/reconstruct={args.reconstruct}", f"hydro/rsolver={args.rsolver}"],
                cwd=folder, stdout=log, stderr=subprocess.STDOUT, check=True,
            )

        reference = [read_stage(path) for path in sorted(folder.glob("stage*.txt"))]
        assert len(reference) == len(coefficients), folder
        (dx, dy, cs, dfloor, dt, _), first = reference[0]
        initial = first["saved"]
        assert dt == 0.001, f"AthenaK took dt={dt}"

        # Both solvers use exactly the reference's initial conserved state and dt.
        step = jax.jit(
            lambda U, dt: isothermal_step(
                U, dt, dx, dy, cs, dfloor, integrator, trace=True,
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
            final.sum(axis=(0, 1)), initial.sum(axis=(0, 1)), rtol=0, atol=2e-12,
        )

        worst = dict.fromkeys(("face states", "face fluxes", "conserved"), 0.0)
        for number, (ref, computed) in enumerate(zip(reference, stages, strict=True), 1):
            metadata, expected = ref
            np.testing.assert_equal(metadata[:5], (dx, dy, cs, dfloor, dt))
            np.testing.assert_equal(metadata[5], coefficients[number - 1][2] * dt)
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
                np.testing.assert_allclose(
                    actual, expected[field], rtol=2e-14, atol=atol,
                    err_msg=f"{case} {integrator} stage {number} {field}",
                )
        print(f"{case:6s} {integrator}  {len(stages)} stages  "
              + "  ".join(f"{name} {error:.3e}" for name, error in worst.items()))

with (args.output / "stage_errors.csv").open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
