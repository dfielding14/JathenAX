"""Check uniform and Smagorinsky viscosity on a periodic sinusoidal shear."""

import argparse
import json
from pathlib import Path

import jax
import numpy as np

from jathenax.hydro import isothermal_timestep
from jathenax.run import evolve_isothermal


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path, help="Directory for snapshots and results")
parser.add_argument("--athenak", type=Path, help="Directory containing the AthenaK runs")
args = parser.parse_args()
args.output.mkdir(parents=True)
jax.config.update("jax_enable_x64", True)

# Match the AthenaK shear benchmark in a unit square.
n, mode, amplitude = 256, 8, 0.5
h, cs, cfl, tlim = 1.0 / n, 1.0, 0.4, 5.0
viscosity, smagorinsky_cs = 1.5e-5, 0.2
x = (np.arange(n) + 0.5) * h
k = 2.0 * np.pi * mode
U = np.zeros((n, n, 3))
U[..., 0] = 1.0
U[..., 2] = amplitude * np.sin(k * x)
U = jax.device_put(U)


def kinetic_energy(U):
    return 0.5 * np.mean(np.sum(U[..., 1:]**2, axis=-1) / U[..., 0])


E0 = kinetic_energy(np.asarray(U))
metrics = {}
for case, nu, Cs in (("uniform", viscosity, 0.0), ("smag", 0.0, smagorinsky_cs)):
    final, time, _, cycles = jax.device_get(evolve_isothermal(
        U, h, h, cs, 1e-10, "rk3", cfl, tlim, 10_000,
        viscosity=nu, smagorinsky_cs=Cs,
    ))
    assert time == tlim, f"{case} stopped at {time}"
    np.testing.assert_allclose(final[..., 0], 1.0, rtol=0, atol=2e-11)
    np.testing.assert_allclose(final[..., 1], 0.0, rtol=0, atol=2e-11)
    np.testing.assert_allclose(final[..., 2].mean(), 0.0, rtol=0, atol=2e-11)

    energy = kinetic_energy(final)
    result = dict(energy=float(energy), energy_ratio=float(energy / E0), cycles=int(cycles))
    if case == "uniform":
        # The exact decay rate on this grid uses the centered Laplacian symbol.
        k_grid_sq = 4.0 * np.sin(k * h / 2.0)**2 / h**2
        result["continuum_energy"] = float(E0 * np.exp(-2.0 * nu * k**2 * tlim))
        result["discrete_energy"] = float(E0 * np.exp(-2.0 * nu * k_grid_sq * tlim))
        np.testing.assert_allclose(energy, result["discrete_energy"], rtol=2e-9)
    else:
        first, _, first_dt, _ = jax.device_get(evolve_isothermal(
            U, h, h, cs, 1e-10, "rk3", cfl, tlim, 1, smagorinsky_cs=Cs,
        ))
        # The continuum mean of |cos(kx)|^3 is 4 / (3*pi).
        loss_rate = (E0 - kinetic_energy(first)) / (first_dt * E0)
        exact_rate = 16.0 / (3.0 * np.pi) * (Cs * h)**2 * amplitude * k**3
        result["initial_fractional_loss_rate"] = float(loss_rate)
        result["analytic_fractional_loss_rate"] = float(exact_rate)
        np.testing.assert_allclose(loss_rate, exact_rate, rtol=0.01)

    if args.athenak is not None:
        history = np.loadtxt(args.athenak / f"athena_{case}" / "shear.hydro.hst")
        assert history[-1, 0] == tlim
        energy_athena = history[-1, 6:9].sum()
        np.testing.assert_allclose(energy, energy_athena, rtol=5e-10)
        result["athenak_energy"] = float(energy_athena)
        result["relative_athenak_error"] = float(abs(energy / energy_athena - 1.0))

    np.savez(args.output / f"{case}.npz", x=x, initial=np.asarray(U), final=final, time=time)
    metrics[case] = result
    print(f"{case}: {cycles} steps, E(t=5)={energy:.16g}", flush=True)


# Turn the coefficients up for two steps so diffusion sets the timestep.
# For this shear, W and U have the same components because rho is one.
max_face_strain = 2.0 * amplitude * np.sin(k * h / 2.0) / h
for case, nu, Cs in (("uniform_limit", 0.01, 0.0), ("smag_limit", 0.0, 8.0)):
    dt = float(isothermal_timestep(
        U, h, h, cs, cfl, np.inf, 0.0, tlim, viscosity=nu, smagorinsky_cs=Cs,
    ))
    if Cs > 0.0:
        nu_max = (Cs * h)**2 * max_face_strain
        expected_dt = cfl * (3.0 / 16.0) * h**2 / (2.0 * nu_max)
    else:
        expected_dt = cfl * 0.25 * h**2 / nu
    np.testing.assert_allclose(dt, expected_dt, rtol=5e-13)

    _, _, second_dt, cycles = jax.device_get(evolve_isothermal(
        U, h, h, cs, 1e-10, "rk3", cfl, tlim, 2,
        viscosity=nu, smagorinsky_cs=Cs,
    ))
    assert cycles == 2
    result = dict(first_dt=dt, second_dt=float(second_dt), expected_dt=float(expected_dt))
    if args.athenak is not None:
        history = np.loadtxt(args.athenak / f"athena_{case}" / "shear.hydro.hst")
        # The first row gives the initial dt. Later cycle rows give the dt just used.
        dt_ref = history[[0, 2], 1]
        np.testing.assert_allclose([dt, second_dt], dt_ref, rtol=5e-13)
        result["athenak_first_dt"], result["athenak_second_dt"] = map(float, dt_ref)
    metrics[case] = result

(args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
