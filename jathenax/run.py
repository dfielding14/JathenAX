"""Run a periodic isothermal example on the selected JAX device."""

import argparse
from functools import partial
from importlib import import_module
from math import isfinite
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .fluxes import RECONSTRUCTIONS, RIEMANN_SOLVERS
from .forcing import cartesian_modes, forcing_basis, modal_coefficients, update_force
from .hydro import isothermal_step, isothermal_timestep, recover_isothermal
from .initial_conditions import kelvin_helmholtz


@partial(jax.jit, static_argnames=(
    "integrator", "viscosity", "smagorinsky_cs", "drag_rate", "dt_update", "tdriv_start", "dedt",
    "reconstruct", "rsolver", "predict_stress",
))
def evolve_isothermal(U, dx, dy, cs, dfloor, integrator, cfl, tlim, nlim, *,
                      viscosity=0.0, smagorinsky_cs=0.0, drag_rate=0.0,
                      forcing=None, dt_update=1.0, tdriv_start=0.0, dedt=0.0,
                      reconstruct="ppm4", rsolver="roe", predict_stress=None, sgs_weights=None):
    """Advance the device-resident periodic grid to the end time or step limit."""
    for name, value in (("viscosity", viscosity), ("smagorinsky_cs", smagorinsky_cs),
                        ("drag_rate", drag_rate), ("dedt", dedt)):
        if not isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if forcing is not None and (not 0.0 < dt_update < np.inf or not isfinite(tdriv_start)):
        raise ValueError("dt_update must be finite and positive and tdriv_start finite")

    def keep_going(state):
        _, time, _, cycle, _, _ = state
        return (time < tlim) & (cycle < nlim)

    def advance(state):
        U, time, previous_dt, cycle, update, raw_force = state
        _, W = recover_isothermal(U, dfloor)
        dt = isothermal_timestep(
            W, dx, dy, cs, cfl, previous_dt, time, tlim,
            viscosity=viscosity, smagorinsky_cs=smagorinsky_cs, drag_rate=drag_rate,
        )
        power = 0.0
        if forcing is not None:
            coefficients, basis = forcing
            update, raw_force = update_force(
                time, update, raw_force, coefficients, basis, dt_update, tdriv_start,
            )
            power = jnp.where(update >= 0, dedt, 0.0)
        U = isothermal_step(
            U, dt, dx, dy, cs, dfloor, integrator,
            viscosity=viscosity, smagorinsky_cs=smagorinsky_cs, drag_rate=drag_rate,
            raw_force=raw_force, dedt=power,
            reconstruct=reconstruct, rsolver=rsolver,
            predict_stress=predict_stress, sgs_weights=sgs_weights,
        )
        return U, time + dt, dt, cycle + 1, update, raw_force

    time = jnp.asarray(0.0, dtype=U.dtype)
    previous_dt = jnp.asarray(jnp.inf, dtype=U.dtype)
    cycle = jnp.asarray(0, dtype=jnp.int32)
    update = jnp.asarray(-1, dtype=jnp.int32)
    raw_force = jnp.zeros_like(U[..., 1:3]) if forcing is not None else None
    state = (U, time, previous_dt, cycle, update, raw_force)
    return jax.lax.while_loop(keep_going, advance, state)[:4]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problem", nargs="?", default="smooth",
                        choices=("smooth", "kh", "turbulence"))
    parser.add_argument("--nx", type=int, help="Number of x cells")
    parser.add_argument("--ny", type=int, help="Number of y cells")
    parser.add_argument("--tlim", type=float, help="Stop time")
    parser.add_argument("--integrator", default="rk3", choices=("rk2", "rk3", "rk4"))
    parser.add_argument("--reconstruct", choices=RECONSTRUCTIONS, default="ppm4",
                        help="Face reconstruction (default: ppm4)")
    parser.add_argument("--rsolver", choices=RIEMANN_SOLVERS, default="roe",
                        help="Riemann solver (default: roe)")
    parser.add_argument("--viscosity", type=float, help="Uniform kinematic viscosity")
    parser.add_argument("--smagorinsky-cs", type=float, default=0.0)
    parser.add_argument("--sgs-model", help="Python module with load_weights and predict_stress")
    parser.add_argument("--sgs-weights", type=Path, help="Checkpoint passed to load_weights")
    parser.add_argument("--output", type=Path, help="Output .npz file")
    args = parser.parse_args()
    if args.sgs_weights is not None and args.sgs_model is None:
        parser.error("--sgs-weights requires --sgs-model")

    # All examples use a periodic unit square and unit sound speed.
    nx, ny, tlim, filename = {
        "smooth": (64, 48, 0.2, "periodic_isothermal.npz"),
        "kh": (128, 128, 3.0, "kelvin_helmholtz.npz"),
        "turbulence": (256, 256, 45.0, "driven_seed1618.npz"),
    }[args.problem]
    nx = nx if args.nx is None else args.nx
    ny = ny if args.ny is None else args.ny
    tlim = tlim if args.tlim is None else args.tlim
    if min(nx, ny) < 6 or not isfinite(tlim) or tlim <= 0:
        parser.error("each grid dimension must be at least six and tlim finite and positive")
    output = args.output or Path(filename)
    if output.suffix != ".npz":
        parser.error("--output must end in .npz")

    driven = args.problem == "turbulence"
    cs, cfl, nlim = 1.0, 0.4, 100_000
    dfloor = 1.17549e-38 if driven else 1.0e-10
    viscosity = 1.5e-6 if driven else 0.0
    if args.viscosity is not None:
        viscosity = args.viscosity
    drag_rate = 0.3333335 if driven else 0.0
    dx, dy = 1.0 / nx, 1.0 / ny
    x = (np.arange(nx) + 0.5) * dx - 0.5
    y = (np.arange(ny) + 0.5) * dy - 0.5
    forcing = None
    dt_update, dedt = 0.0075, 0.0
    if driven:
        # AthenaK turb.cpp starts with unit density and zero momentum.
        rseed, tcorr, dedt = 1618, 0.75, 0.03703705
        nlow, nhigh, npeak = 3, 5, 4
        U_initial = jnp.zeros((ny, nx, 3)).at[..., 0].set(1.0)
        modes = cartesian_modes(nlow, nhigh, (-nhigh, nhigh), (-nhigh, nhigh))
        updates = int(np.ceil(tlim / dt_update)) + 1
        coefficients = modal_coefficients(
            modes, (1.0, 1.0), nlow, nhigh, npeak, 1.0, rseed, tcorr, dt_update, updates,
        )
        basis = forcing_basis(jnp.asarray(modes), jnp.asarray(x), jnp.asarray(y), (1.0, 1.0))
        forcing = (jnp.asarray(coefficients, dtype=U_initial.dtype), basis)
    elif args.problem == "kh":
        U_initial = jnp.asarray(kelvin_helmholtz(x, y))
    else:
        phase_x, phase_y = np.meshgrid(2.0 * np.pi * x, 2.0 * np.pi * y)
        rho = 1.0 + 0.1 * np.sin(phase_x + phase_y)
        vx = 0.2 * np.cos(phase_y)
        vy = 0.2 * np.sin(phase_x)
        U_initial = jax.device_put(np.stack((rho, rho * vx, rho * vy), axis=-1))

    predict_stress, sgs_weights = None, None
    if args.sgs_model:
        model = import_module(args.sgs_model)
        predict_stress = model.predict_stress
        sgs_weights = jax.device_put(model.load_weights(args.sgs_weights), U_initial.device)

    U, time, dt, cycle = jax.device_get(evolve_isothermal(
        U_initial, dx, dy, cs, dfloor, args.integrator, cfl, tlim, nlim,
        viscosity=viscosity, smagorinsky_cs=args.smagorinsky_cs, drag_rate=drag_rate,
        forcing=forcing, dt_update=dt_update, dedt=dedt,
        reconstruct=args.reconstruct, rsolver=args.rsolver,
        predict_stress=predict_stress, sgs_weights=sgs_weights,
    ))
    if not np.isfinite(U).all() or time != np.asarray(tlim, dtype=time.dtype):
        raise RuntimeError(f"Periodic run ended at t={time} after {cycle} steps")

    initial = np.asarray(U_initial)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, x=x, y=y, initial=initial, conserved=U, time=time, dt=dt, cycle=cycle,
             problem=args.problem, sgs_model=args.sgs_model or "")
    drift = (U.sum(axis=(0, 1), dtype=np.float64)
             - initial.sum(axis=(0, 1), dtype=np.float64)) * dx * dy
    print(f"{args.integrator.upper()} on {U_initial.device}, {U.dtype}, {cycle} steps to t={time:.6g}")
    print(f"Reconstruction {args.reconstruct}, Riemann solver {args.rsolver}")
    print(f"Change in box mass and momenta {drift[:3]}")
    if U.shape[-1] > 3:
        print(f"Change in integrated tracer {drift[3:]}")
    print(output.resolve())
