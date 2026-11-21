"""Run a periodic hydrodynamics example on the selected JAX device."""

import argparse
import json
from functools import partial
from importlib import import_module
from math import isfinite
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .fluxes import RECONSTRUCTIONS, RIEMANN_SOLVERS
from .forcing import cartesian_modes, modal_coefficients, periodic_basis, render_force, update_force
from .hydro import hydro_step, hydro_timestep, recover_primitives
from .input import PROBLEMS, read_input
from .initial_conditions import initial_state


@partial(jax.jit, static_argnames=(
    "eos", "gamma", "tau_cool", "cooling_ramp_time", "cooling_ramp_exponent",
    "integrator", "viscosity", "smagorinsky_cs", "drag_rate",
    "dt_update", "tdriv_start", "dedt", "reconstruct", "rsolver", "predict_stress",
))
def evolve(U, spacing, cfl, tlim, nlim, *, eos="ideal", gamma=5/3, cs=1.0,
           dfloor=1e-10, pfloor=1e-10, integrator="rk3", viscosity=0.0, smagorinsky_cs=0.0,
           drag_rate=0.0, tau_cool=0.0, cooling_ramp_time=0.0, cooling_ramp_exponent=1.0,
           forcing=None, dt_update=1.0, tdriv_start=0.0, dedt=0.0, reconstruct="ppm4",
           rsolver="roe", predict_stress=None, sgs_weights=None):
    """Advance a periodic grid on device to the end time or step limit."""
    if eos == "ideal" and not 1.0 < gamma < np.inf:
        raise ValueError("an ideal gas requires finite gamma > 1")
    for name, value in (("viscosity", viscosity), ("smagorinsky_cs", smagorinsky_cs),
                        ("drag_rate", drag_rate), ("dedt", dedt), ("tau_cool", tau_cool)):
        if not isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if forcing is not None and (not 0.0 < dt_update < np.inf or not isfinite(tdriv_start)):
        raise ValueError("dt_update must be finite and positive and tdriv_start finite")

    def keep_going(state):
        _, time, _, cycle, _, _ = state
        return (time < tlim) & (cycle < nlim)

    def advance(state):
        U, time, previous_dt, cycle, update, raw_force = state
        _, W = recover_primitives(U, eos, gamma, dfloor, pfloor)
        dt = hydro_timestep(
            W, spacing, cfl, previous_dt, time, tlim, eos=eos, gamma=gamma, cs=cs,
            tau_cool=tau_cool,
            viscosity=viscosity, smagorinsky_cs=smagorinsky_cs, drag_rate=drag_rate,
        )
        power = 0.0
        if forcing is not None:
            coefficients, basis = forcing
            update, raw_force = update_force(
                time, update, raw_force, coefficients, basis, dt_update, tdriv_start,
            )
            power = jnp.where(update >= 0, dedt, 0.0)
        U = hydro_step(
            U, dt, spacing, eos=eos, gamma=gamma, cs=cs, dfloor=dfloor, pfloor=pfloor,
            integrator=integrator, tau_cool=tau_cool, cooling_ramp_time=cooling_ramp_time,
            cooling_ramp_exponent=cooling_ramp_exponent, time=time,
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
    raw_force = jnp.zeros_like(U[..., 1:U.ndim]) if forcing is not None else None
    state = (U, time, previous_dt, cycle, update, raw_force)
    return jax.lax.while_loop(keep_going, advance, state)[:4]


def evolve_isothermal(U, dx, dy, cs, dfloor, integrator, cfl, tlim, nlim, **kwargs):
    return evolve(U, (dx, dy), cfl, tlim, nlim, eos="isothermal", cs=cs, dfloor=dfloor,
                  integrator=integrator, **kwargs)


def prepare_forcing(driving, coords, tlim, dtype):
    """Build the seeded OU sequence and separable phase factors for the unit box."""
    dt_update, start, dedt = (driving[k] for k in ("dt_update", "tdriv_start", "dedt"))
    if not 0 < dt_update < np.inf or not isfinite(start):
        raise ValueError("forcing requires finite tdriv_start and positive finite dt_update")
    if not isfinite(dedt) or dedt < 0:
        raise ValueError("dedt must be finite and nonnegative")
    if dedt == 0 or start >= tlim:
        return None
    bounds = [(driving[f"min_k{k}"], driving[f"max_k{k}"]) for k in "xyz"[:len(coords)]]
    nlow, nhigh = driving["nlow"], driving["nhigh"]
    lengths = (1.0,) * len(coords)
    updates = int(np.ceil((tlim - start) / dt_update)) + 1
    modes = cartesian_modes(nlow, nhigh, *bounds)
    coefficients = modal_coefficients(
        modes, lengths, nlow, nhigh, driving["npeak"], driving["sol_fraction"],
        driving["rseed"], driving["tcorr"], dt_update, updates,
    )
    basis = periodic_basis(jnp.asarray(modes), tuple(jnp.asarray(x, dtype) for x in coords), lengths)
    return jnp.asarray(coefficients, dtype=dtype), basis


def initial_state_from_deck(settings, coordinates):
    """Prepare conserved fields, optionally drawing turbulent velocities independently of the driver."""
    problem, hydro = settings["problem"], settings["hydro"]
    u = initial_state(problem["name"], coordinates, hydro["eos"], hydro["gamma"],
                      hydro["iso_sound_speed"], problem.get("axis", "x"),
                      initial_sound_speed=problem.get("initial_sound_speed", 1.0))
    target = problem.get("initial_vrms", 0.0)
    if target == 0:
        return u
    if not isfinite(target) or target < 0 or problem["initial_velocity_seed"] <= 0:
        raise ValueError("turbulent initialization needs a finite nonnegative rms speed and a positive seed")
    # Draw the first field with the driving spectrum and projection, on a separate RNG stream.
    draw = dict(settings["turb_driving"], rseed=problem["initial_velocity_seed"],
                tdriv_start=0., dedt=1.)
    coeff, basis = prepare_forcing(draw, coordinates, draw["dt_update"], jnp.asarray(u).dtype)
    v = np.asarray(render_force(coeff[0], basis), dtype=float)
    rho = u[..., 0]
    axes = tuple(range(rho.ndim))
    v = v - np.mean(rho[..., None]*v, axis=axes)/rho.mean()
    rms = np.sqrt(np.mean(rho*np.sum(v*v, axis=-1))/rho.mean())
    if not np.isfinite(rms) or rms <= 0:
        raise ValueError("the initial velocity field has no power on this mesh")
    v *= target/rms
    u[..., 1:1+rho.ndim] = rho[..., None]*v
    if hydro["eos"] == "ideal":
        u[..., 1+rho.ndim] += 0.5*rho*np.sum(v*v, axis=-1)
    return u


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problem", nargs="?", choices=PROBLEMS)
    parser.add_argument("-i", "--input", type=Path, help="TOML input deck")
    parser.add_argument("--nx", type=int, help="Number of x cells")
    parser.add_argument("--ny", type=int, help="Number of y cells")
    parser.add_argument("--nz", type=int, help="Number of z cells (1 disables z)")
    parser.add_argument("--eos", choices=("isothermal", "ideal"))
    parser.add_argument("--gamma", type=float, help="Ideal-gas adiabatic index")
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
        settings = read_input(args.input, args.problem, mesh_size=(args.nx, args.ny, args.nz))
    except (OSError, ValueError) as error:
        parser.error(str(error))

    mesh, time, hydro = (settings[key] for key in ("mesh", "time", "hydro"))
    hydro_keys = ("eos", "gamma", "reconstruct", "rsolver", "viscosity", "smagorinsky_cs")
    for section, keys in ((time, ("tlim", "integrator")), (hydro, hydro_keys)):
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

    nx, ny, nz = (mesh[k] for k in ("nx", "ny", "nz"))
    if nx < 6 or (ny != 1 and ny < 6) or (nz != 1 and nz < 6) or (nz > 1 and ny == 1):
        parser.error("active grid sizes must be at least six, in x, y, z order (1 disables y or z)")
    sizes = tuple(n for n in (nx, ny, nz) if n > 1)
    ndim = len(sizes)
    eos, gamma, pfloor = hydro["eos"], hydro["gamma"], hydro["pfloor"]
    tlim, nlim, cfl = time["tlim"], time["nlim"], time["cfl_number"]
    integrator = time["integrator"]
    reconstruct, rsolver = hydro["reconstruct"], hydro["rsolver"]
    cs, dfloor = hydro["iso_sound_speed"], hydro["dfloor"]
    viscosity, smagorinsky_cs = hydro["viscosity"], hydro["smagorinsky_cs"]
    drag_rate = settings["hydro_srcterms"]["drag_rate"]
    tau_cool = settings["hydro_srcterms"]["tau_cool"]
    if eos not in ("ideal", "isothermal") or (eos == "ideal" and not 1 < gamma < np.inf):
        parser.error("hydro.eos must be isothermal or ideal, with finite gamma > 1 for ideal gas")
    if tau_cool < 0 or not isfinite(tau_cool) or (tau_cool and eos != "ideal"):
        parser.error("positive finite tau_cool requires ideal gas, zero disables cooling")
    if model_name and (ndim != 2 or eos != "isothermal"):
        parser.error("the learned stress interface requires 2D isothermal hydro")
    if not 1 <= nlim <= np.iinfo(np.int32).max:
        parser.error("time.nlim must be a positive 32-bit cycle limit")
    for key, value in (("time.tlim", tlim), ("time.cfl_number", cfl),
                       ("hydro.iso_sound_speed", cs), ("hydro.dfloor", dfloor), ("hydro.pfloor", pfloor)):
        if not isfinite(value) or value <= 0:
            parser.error(f"{key} must be finite and positive")
    if integrator not in ("rk2", "rk3", "rk4"):
        parser.error("time.integrator must be rk2, rk3, or rk4")
    if reconstruct not in RECONSTRUCTIONS or rsolver not in RIEMANN_SOLVERS:
        parser.error("invalid hydro.reconstruct or hydro.rsolver")
    if rsolver == "hllc" and eos != "ideal":
        parser.error("HLLC requires an ideal-gas EOS")
    output = args.output or Path(settings["output"]["file"])
    if output.suffix != ".npz":
        parser.error("--output must end in .npz")

    spacing = tuple(1.0 / n for n in sizes)
    coords = tuple((np.arange(n) + 0.5) * h - 0.5 for n, h in zip(sizes, spacing))
    try:
        initial = initial_state_from_deck(settings, coords)
    except ValueError as error:
        parser.error(str(error))
    U_initial = jnp.asarray(initial)
    forcing = None
    dt_update, tdriv_start, dedt = 1.0, 0.0, 0.0
    if problem == "turbulence":
        driving = settings["turb_driving"]
        dt_update, tdriv_start, dedt = (driving[k] for k in ("dt_update", "tdriv_start", "dedt"))
        try:
            forcing = prepare_forcing(driving, coords, tlim, U_initial.dtype)
        except ValueError as error:
            parser.error(str(error))

    predict_stress, sgs_weights = None, None
    if model_name:
        model = import_module(model_name)
        predict_stress = model.predict_stress
        sgs_weights = jax.device_put(model.load_weights(checkpoint), U_initial.device)

    U, time, dt, cycle = jax.device_get(evolve(
        U_initial, spacing, cfl, tlim, nlim, eos=eos, gamma=gamma, cs=cs, dfloor=dfloor,
        pfloor=pfloor, integrator=integrator, tau_cool=tau_cool,
        cooling_ramp_time=settings["hydro_srcterms"]["cooling_ramp_time"],
        cooling_ramp_exponent=settings["hydro_srcterms"]["cooling_ramp_exponent"],
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
    settings["output"]["file"] = str(output)
    settings["sgs"] = dict(model=model_name, weights=str(checkpoint) if checkpoint else "")
    np.savez(output, **dict(zip("xyz", coords)), initial=initial, conserved=U, time=time,
             dt=dt, cycle=cycle, problem=problem, sgs_model=model_name, eos=eos, gamma=gamma,
             settings=json.dumps(settings))
    axes = tuple(range(ndim))
    drift = U.mean(axis=axes, dtype=np.float64) - initial.mean(axis=axes, dtype=np.float64)
    print(f"{integrator.upper()} on {U_initial.device}, {U.dtype}, {cycle} steps to t={time:.6g}")
    if not reached_tlim:
        print(f"Reached time.nlim={nlim} before tlim={tlim:g}")
    print(f"Reconstruction {reconstruct}, Riemann solver {rsolver}")
    print(f"Change in box mass and momenta {drift[:ndim + 1]}")
    ngas = ndim + 1 + (eos == "ideal")
    if eos == "ideal":
        print(f"Change in box total energy {drift[ndim + 1]:.10g}")
        if problem == "turbulence":
            rho = U[..., 0].astype(np.float64)
            momenta = [U[..., i + 1].astype(np.float64) for i in range(ndim)]
            kinetic = 0.5 * sum(np.mean(m*m/rho) for m in momenta)
            bulk = 0.5 * sum(m.mean()**2 for m in momenta) / rho.mean()
            thermal = U[..., ndim + 1].mean(dtype=np.float64) - kinetic
            mach = np.sqrt(2.0 * max(0.0, kinetic - bulk) / (gamma * (gamma - 1.0) * thermal))
            print(f"Final box turbulent Mach number {mach:.6g}")
    if U.shape[-1] > ngas:
        print(f"Change in integrated tracer {drift[ngas:]}")
    print(output.resolve())
