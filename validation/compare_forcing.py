"""Compare JathenAX to the full-precision AthenaK seed-1618 forcing trace."""

import argparse
from configparser import ConfigParser
import csv
from itertools import islice
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jathenax.forcing import (
    cartesian_modes, forcing_basis, modal_coefficients, ran_gaussian_st, ran_st, update_force,
)
from jathenax.hydro import isothermal_step, isothermal_timestep, recover_isothermal
from jathenax.run import evolve_isothermal


def compare(case, cycle, field, actual, expected, *, atol=0.0, rtol=0.0):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.dtype == expected.dtype == np.float64
    a, b = actual.view(np.uint64), expected.view(np.uint64)
    different = np.flatnonzero(a != b)
    sign = np.uint64(1 << 63)
    a_ordered = np.where(a & sign, ~a, a | sign)
    b_ordered = np.where(b & sign, ~b, b | sign)
    rows.append(dict(
        case=case, cycle=cycle, field=field, max_abs_error=np.max(np.abs(actual - expected)),
        max_ulps=np.max(np.maximum(a_ordered, b_ordered) - np.minimum(a_ordered, b_ordered)),
        different_values=len(different), total_values=actual.size,
        first_different_flat_index=int(different[0]) if len(different) else "",
    ))
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol,
                               err_msg=f"{case} cycle {cycle} {field}")


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("reference", type=Path, help="Directory of AthenaK trace dumps")
parser.add_argument("output", type=Path, help="New directory for comparison outputs")
args = parser.parse_args()
args.output.mkdir(parents=True)
jax.config.update("jax_enable_x64", True)
deck = ConfigParser()
deck.read_string(Path(__file__).with_name("seed1618.athinput").read_text()
                 .replace("<", "[").replace(">", "]"))

mesh, time_params, hydro, driving = (deck[s] for s in ("mesh", "time", "hydro", "turb_driving"))
nx, ny = mesh.getint("nx1"), mesh.getint("nx2")
Lx = mesh.getfloat("x1max") - mesh.getfloat("x1min")
Ly = mesh.getfloat("x2max") - mesh.getfloat("x2min")
dx, dy = Lx / nx, Ly / ny
x = mesh.getfloat("x1min") + (np.arange(nx) + 0.5) * dx
y = mesh.getfloat("x2min") + (np.arange(ny) + 0.5) * dy
cs, dfloor = hydro.getfloat("iso_sound_speed"), hydro.getfloat("dfloor")
cfl, tlim = time_params.getfloat("cfl_number"), time_params.getfloat("tlim")
integrator = time_params["integrator"]
transport = dict(viscosity=hydro.getfloat("viscosity"),
                 drag_rate=deck["hydro_srcterms"].getfloat("drag_rate"))
seed = driving.getint("rseed")
dt_update, start, dedt = [driving.getfloat(key) for key in ("dt_update", "tdriv_start", "dedt")]
nlow, nhigh = driving.getint("nlow"), driving.getint("nhigh")
kx_bounds = (driving.getint("min_kx"), driving.getint("max_kx"))
ky_bounds = (driving.getint("min_ky"), driving.getint("max_ky"))
modes = cartesian_modes(nlow, nhigh, kx_bounds, ky_bounds)

rows = []
for field, rng, tolerance in (
    ("uniform", ran_st, 0.0), ("gaussian", ran_gaussian_st, 2e-15),
):
    reference = np.loadtxt(args.reference / f"{field}.txt").ravel()
    actual = np.fromiter(islice(rng(seed), reference.size), dtype=float)
    compare("setup", -1, field, actual, reference, atol=tolerance)
    print(f"{field}: {reference.size} draws, max difference {rows[-1]['max_abs_error']:.3e}")
    np.save(args.output / f"{field}.npy", actual)

reference_modes = np.loadtxt(args.reference / "modes.txt").reshape(-1, len(modes), 7)
coefficients = modal_coefficients(
    modes, (Lx, Ly), nlow, nhigh, driving.getfloat("npeak"), driving.getfloat("sol_fraction"),
    seed, driving.getfloat("tcorr"), dt_update, len(reference_modes),
)
for update, reference in enumerate(reference_modes):
    np.testing.assert_equal(reference[:, 0], np.full(len(modes), update))
    np.testing.assert_array_equal(modes, reference[:, 1:3])
    compare("setup", update, "modal_coefficients", coefficients[update].reshape(-1, 4),
            reference[:, 3:], atol=3e-15)
np.savez(args.output / "modes.npz", modes=modes, coefficients=coefficients)

steps = np.loadtxt(args.reference / "steps.txt")
assert len(steps) == time_params.getint("nlim")
assert np.array_equal(np.unique(steps[:, 3]), [0, 1, 2])
basis = forcing_basis(jnp.asarray(modes), jnp.asarray(x), jnp.asarray(y), (Lx, Ly))
coefficients = jnp.asarray(coefficients)
initial = jnp.zeros((ny, nx, 3)).at[..., 0].set(deck["problem"].getfloat("d_n"))


@jax.jit
def advance(U, time, previous_dt, update, raw_force, reference_dt, fixed_dt):
    _, W = recover_isothermal(U, dfloor)
    selected_dt = isothermal_timestep(W, dx, dy, cs, cfl, previous_dt, time, tlim, **transport)
    dt = jnp.where(fixed_dt, reference_dt, selected_dt)
    update, raw_force = update_force(time, update, raw_force, coefficients, basis, dt_update, start)
    U, stages = isothermal_step(
        U, dt, dx, dy, cs, dfloor, integrator, **transport,
        raw_force=raw_force, dedt=dedt, trace=True,
    )
    final_stage = stages[-1]
    return U, dt, update, raw_force, final_stage["before_kick"], final_stage["acceleration"]


for case in ("fixed", "adaptive"):
    U, time, previous_dt = initial, 0.0, np.inf
    update, raw_force = jnp.int32(-1), jnp.zeros_like(U[..., 1:])
    for cycle, start_time, reference_dt, reference_update in steps:
        cycle = int(cycle)
        compare(case, cycle, "time", time, start_time, rtol=2e-14)
        U, dt, update, raw_force, before, acceleration = advance(
            U, time, previous_dt, update, raw_force, reference_dt, case == "fixed",
        )
        dt, update_host, raw, before, acceleration, after = jax.device_get(
            (dt, update, raw_force, before, acceleration, U),
        )
        assert update_host == reference_update
        compare(case, cycle, "dt", dt, reference_dt, rtol=2e-14)
        fields = dict(raw=raw, before=before, acceleration=acceleration, after=after)
        for field, actual in fields.items():
            reference = np.fromfile(args.reference / f"{field}_{cycle}.bin").reshape(actual.shape)
            compare(case, cycle, field, actual, reference, atol=5e-14, rtol=2e-14)
        momentum_added = after[..., 1:] - before[..., 1:]
        np.testing.assert_allclose(momentum_added.mean(axis=(0, 1)), 0.0, atol=5e-16)
        np.testing.assert_array_equal(after[..., 0], before[..., 0])
        energy_added = np.mean(
            0.5 * np.sum(momentum_added * (after[..., 1:] + before[..., 1:]), axis=-1)
            / before[..., 0]
        )
        np.testing.assert_allclose(energy_added, dedt * dt, rtol=2e-13)
        np.savez(args.output / f"{case}_{cycle:02d}.npz", time=time, dt=dt, update=update_host,
                 energy_added=energy_added, **fields)
        time, previous_dt = time + dt, dt
    print(f"{case}: {len(steps)} cycles to t={time:.17g}")

# Compare the production while loop to the separately traced adaptive trajectory.
final, final_time, final_dt, cycles = jax.device_get(evolve_isothermal(
    initial, dx, dy, cs, dfloor, integrator, cfl, tlim, len(steps), **transport,
    forcing=(coefficients, basis), dt_update=dt_update, tdriv_start=start, dedt=dedt,
))
assert cycles == len(steps)
compare("production", int(cycles), "conserved", final, after, atol=5e-14)
compare("production", int(cycles), "time", final_time, time, rtol=2e-14)
compare("production", int(cycles), "dt", final_dt, previous_dt, rtol=2e-14)

with (args.output / "errors.csv").open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
