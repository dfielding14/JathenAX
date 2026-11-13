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
from .input import PROBLEMS, read_input
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
    parser.add_argument("problem", nargs="?", choices=PROBLEMS)
    parser.add_argument("-i", "--input", type=Path, help="TOML input deck")
    parser.add_argument("--nx", type=int, help="Number of x cells")
    parser.add_argument("--ny", type=int, help="Number of y cells")
    parser.add_argument("--tlim", type=float, help="Stop time")
    parser.add_argument("--integrator", choices=("rk2", "rk3", "rk4"))
    parser.add_argument("--reconstruct", choices=RECONSTRUCTIONS,
                        help="Face reconstruction (default: ppm4)")
    parser.add_argument("--rsolver", choices=RIEMANN_SOLVERS,
                        help="Riemann solver (default: roe)")
    parser.add_argument("--viscosity", type=float, help="Uniform kinematic viscosity")
    parser.add_argument("--smagorinsky-cs", type=float)
    parser.add_argument("--sgs-model", help="Python module with load_weights and predict_stress")
    parser.add_argument("--sgs-weights", type=Path, help="Checkpoint passed to load_weights")
    parser.add_argument("--output", type=Path, help="Output .npz file")
    args = parser.parse_args()
    try:
        settings = read_input(args.input, args.problem)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    mesh, time, hydro = (settings[key] for key in ("mesh", "time", "hydro"))
    for section, keys in ((mesh, ("nx", "ny")), (time, ("tlim", "integrator")),
                          (hydro, ("reconstruct", "rsolver", "viscosity", "smagorinsky_cs"))):
        for key in keys:
            value = getattr(args, key)
            if value is not None:
                section[key] = value
    problem = settings["problem"]["name"]
    model_name = args.sgs_model if args.sgs_model is not None else settings["sgs"]["model"]
    checkpoint = args.sgs_weights
    if checkpoint is None and settings["sgs"]["weights"]:
        checkpoint = Path(settings["sgs"]["weights"])
    if checkpoint is not None and not model_name:
        parser.error("SGS weights require a model (sgs.model or --sgs-model)")

    nx, ny = mesh["nx"], mesh["ny"]
    tlim, nlim, cfl = time["tlim"], time["nlim"], time["cfl_number"]
    integrator = time["integrator"]
    reconstruct, rsolver = hydro["reconstruct"], hydro["rsolver"]
    cs, dfloor = hydro["iso_sound_speed"], hydro["dfloor"]
    viscosity, smagorinsky_cs = hydro["viscosity"], hydro["smagorinsky_cs"]
    drag_rate = settings["hydro_srcterms"]["drag_rate"]
    if min(nx, ny) < 6:
        parser.error("each grid dimension must be at least six")
    if not 1 <= nlim <= np.iinfo(np.int32).max:
        parser.error("time.nlim must be a positive 32-bit cycle limit")
    for key, value in (("time.tlim", tlim), ("time.cfl_number", cfl),
                       ("hydro.iso_sound_speed", cs), ("hydro.dfloor", dfloor)):
        if not isfinite(value) or value <= 0:
            parser.error(f"{key} must be finite and positive")
    if integrator not in ("rk2", "rk3", "rk4"):
        parser.error("time.integrator must be rk2, rk3, or rk4")
    if reconstruct not in RECONSTRUCTIONS or rsolver not in RIEMANN_SOLVERS:
        parser.error("invalid hydro.reconstruct or hydro.rsolver")
    output = args.output or Path(settings["output"]["file"])
    if output.suffix != ".npz":
        parser.error("--output must end in .npz")

    dx, dy = 1.0 / nx, 1.0 / ny
    x = (np.arange(nx) + 0.5) * dx - 0.5
    y = (np.arange(ny) + 0.5) * dy - 0.5
    forcing = None
    dt_update, tdriv_start, dedt = 1.0, 0.0, 0.0
    if problem == "turbulence":
        # AthenaK turb.cpp starts with unit density and zero momentum.
        driving = settings["turb_driving"]
        dt_update, tdriv_start, dedt = (driving[k] for k in ("dt_update", "tdriv_start", "dedt"))
        if not (0 < dt_update < np.inf and isfinite(tdriv_start)):
            parser.error("forcing requires finite tdriv_start and positive finite dt_update")
        U_initial = jnp.zeros((ny, nx, 3)).at[..., 0].set(1.0)
        nlow, nhigh = driving["nlow"], driving["nhigh"]
        kx_bounds = driving["min_kx"], driving["max_kx"]
        ky_bounds = driving["min_ky"], driving["max_ky"]
        modes = cartesian_modes(nlow, nhigh, kx_bounds, ky_bounds)
        updates = int(np.ceil(max(0.0, tlim - tdriv_start) / dt_update)) + 1
        coefficients = modal_coefficients(
            modes, (1.0, 1.0), nlow, nhigh, driving["npeak"], driving["sol_fraction"],
            driving["rseed"], driving["tcorr"], dt_update, updates,
        )
        basis = forcing_basis(jnp.asarray(modes), jnp.asarray(x), jnp.asarray(y), (1.0, 1.0))
        forcing = (jnp.asarray(coefficients, dtype=U_initial.dtype), basis)
    elif problem == "kh":
        U_initial = jnp.asarray(kelvin_helmholtz(x, y))
    else:
        phase_x, phase_y = np.meshgrid(2.0 * np.pi * x, 2.0 * np.pi * y)
        rho = 1.0 + 0.1 * np.sin(phase_x + phase_y)
        vx = 0.2 * np.cos(phase_y)
        vy = 0.2 * np.sin(phase_x)
        U_initial = jax.device_put(np.stack((rho, rho * vx, rho * vy), axis=-1))

    predict_stress, sgs_weights = None, None
    if model_name:
        model = import_module(model_name)
        predict_stress = model.predict_stress
        sgs_weights = jax.device_put(model.load_weights(checkpoint), U_initial.device)

    U, time, dt, cycle = jax.device_get(evolve_isothermal(
        U_initial, dx, dy, cs, dfloor, integrator, cfl, tlim, nlim,
        viscosity=viscosity, smagorinsky_cs=smagorinsky_cs, drag_rate=drag_rate,
        forcing=forcing, dt_update=dt_update, tdriv_start=tdriv_start, dedt=dedt,
        reconstruct=reconstruct, rsolver=rsolver,
        predict_stress=predict_stress, sgs_weights=sgs_weights,
    ))
    reached_tlim = time == np.asarray(tlim, dtype=time.dtype)
    if not np.isfinite(U).all() or not np.isfinite(time) or (not reached_tlim and cycle != nlim):
        raise RuntimeError(f"Periodic run ended at t={time} after {cycle} steps")

    initial = np.asarray(U_initial)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, x=x, y=y, initial=initial, conserved=U, time=time, dt=dt, cycle=cycle,
             problem=problem, sgs_model=model_name)
    drift = (U.sum(axis=(0, 1), dtype=np.float64)
             - initial.sum(axis=(0, 1), dtype=np.float64)) * dx * dy
    print(f"{integrator.upper()} on {U_initial.device}, {U.dtype}, {cycle} steps to t={time:.6g}")
    if not reached_tlim:
        print(f"Reached time.nlim={nlim} before tlim={tlim:g}")
    print(f"Reconstruction {reconstruct}, Riemann solver {rsolver}")
    print(f"Change in box mass and momenta {drift[:3]}")
    if U.shape[-1] > 3:
        print(f"Change in integrated tracer {drift[3:]}")
    print(output.resolve())
