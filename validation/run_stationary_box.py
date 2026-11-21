"""Run the periodic ideal-gas pilot with a per-step Mach and energy history."""

import argparse
from functools import partial
import json
import os
from pathlib import Path
import shutil
from time import monotonic

import jax
import jax.numpy as jnp
import numpy as np

from jathenax.forcing import update_force
from jathenax.hydro import hydro_step, hydro_timestep, recover_primitives
from jathenax.input import read_input
from jathenax.run import initial_state_from_deck, prepare_forcing
from jathenax.spectra import mode_shells, shell_energy
from validation.replay_window import step_record


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("deck", type=Path)
parser.add_argument("--replay-at", nargs="+", type=float, default=(), metavar="TIME",
                    help="Start a filtered window on the first fine step at or after each time")
parser.add_argument("--replay-factor", type=int, default=2, help="Fine cells per coarse cell along each axis")
parser.add_argument("--replay-steps", type=int, default=4, help="Consecutive fine steps per window")
parser.add_argument("--spectrum-dt", type=float, help="Save velocity spectra on the first step at or after each time interval")
args = parser.parse_args()
settings = read_input(args.deck)
problem, mesh, hydro, time, sources = (settings[k] for k in (
    "problem", "mesh", "hydro", "time", "hydro_srcterms"))
sizes = tuple(mesh[k] for k in ("nx", "ny", "nz"))
if (problem["name"] != "turbulence" or hydro["eos"] != "ideal" or min(sizes) < 6 or
        time["integrator"] != "rk3" or sources["drag_rate"] != 0 or settings["sgs"]["model"]):
    parser.error("this budget measures the periodic 3D gas-only RK3 pilot with zero drag")
driving = settings["turb_driving"]
tau, power, start = sources["tau_cool"], driving["dedt"], driving["tdriv_start"]
if tau <= 0 or power <= 0 or start != 0:
    parser.error("the pilot needs positive cooling time and driving power from t=0")
gamma, dfloor, pfloor = (hydro[k] for k in ("gamma", "dfloor", "pfloor"))
spacing = tuple(1/n for n in sizes)
coords = tuple((np.arange(n) + 0.5) / n - 0.5 for n in sizes)
jax.config.update("jax_enable_x64", jax.default_backend().lower() != "metal")
initial = initial_state_from_deck(settings, coords)
U = jnp.asarray(initial)
initial = initial.astype(U.dtype, copy=False)
tlim, cfl = time["tlim"], time["cfl_number"]
if args.replay_at:
    if args.replay_factor < 1 or any(n % args.replay_factor for n in sizes):
        parser.error("the replay factor must be positive and divide each grid dimension")
    if args.replay_steps < 1 or any(not 0 <= t < tlim for t in args.replay_at):
        parser.error("replay requires a positive step count and start times in [0, tlim)")
out = Path(settings["output"]["file"])
history = out.with_name("history.csv")
spectrum_path = out.with_name("velocity_spectra.npz")
if args.spectrum_dt is not None and not 0 < args.spectrum_dt < np.inf:
    parser.error("--spectrum-dt must be finite and positive")
if history.exists() or out.exists() or (args.spectrum_dt and spectrum_path.exists()):
    parser.error(f"this run already has output in {out.parent}")
out.parent.mkdir(parents=True, exist_ok=True)
if args.deck.resolve() != out.with_name("input.toml").resolve():
    shutil.copyfile(args.deck, out.with_name("input.toml"))
print(f"Preparing seed {driving['rseed']} on {U.device} ({U.dtype}) for t={tlim:g}", flush=True)
forcing = prepare_forcing(driving, coords, tlim, U.dtype)


def box_means(U):
    gas = jnp.mean(U, axis=(0, 1, 2))
    kinetic = 0.5 * jnp.mean(jnp.sum(U[..., 1:4]**2, axis=-1) / U[..., 0])
    return jnp.concatenate((gas, jnp.stack((kinetic, gas[4] - kinetic))))


def velocity_spectrum(U, shell):
    gas = np.asarray(U, dtype=float)
    v = gas[..., 1:4]/gas[..., :1]
    v -= v.mean(axis=(0, 1, 2))
    spectrum = shell_energy(np.moveaxis(v, -1, 0), shell)
    np.testing.assert_allclose(spectrum.sum(), 0.5*np.mean(np.sum(v*v, axis=-1)), rtol=1e-12)
    return spectrum


@partial(jax.jit, donate_argnums=(0, 4), static_argnames=("record",))
def advance(U, t, previous_dt, index, raw, forcing, remaining, *, record):
    _, W = recover_primitives(U, "ideal", gamma, dfloor, pfloor)
    transport = dict(viscosity=hydro["viscosity"], smagorinsky_cs=hydro["smagorinsky_cs"])
    dt = hydro_timestep(W, spacing, cfl, previous_dt, 0.0, remaining, gamma=gamma, tau_cool=tau,
                        **transport)
    index, raw = update_force(t, index, raw, *forcing, driving["dt_update"], start)
    initial_state_at_step = U
    U, stages = hydro_step(U, dt, spacing, gamma=gamma, dfloor=dfloor, pfloor=pfloor,
                           reconstruct=hydro["reconstruct"], rsolver=hydro["rsolver"],
                           raw_force=raw, dedt=power, tau_cool=tau, time=t,
                           cooling_ramp_time=sources["cooling_ramp_time"],
                           cooling_ramp_exponent=sources["cooling_ramp_exponent"], trace=True, **transport)
    # Independently measure the RK3 thermal sink and the finite kinetic work of the kick.
    loss = dt/tau * sum(w * s["cooling_fraction"] * s["primitives"][..., 4].mean()
                        for w, s in zip((1/6, 1/6, 2/3), stages))
    pre = stages[-1]["before_kick"]
    dv = dt * stages[-1]["acceleration"]
    work = jnp.mean(jnp.sum((pre[..., 1:4] + 0.5*pre[..., :1]*dv) * dv, axis=-1))
    min_rho, min_thermal = jnp.inf, jnp.inf
    for stage in stages:
        u = stage["conserved"]  # Before conservative-to-primitive floors.
        rho = u[..., 0]
        e = u[..., 4] - 0.5*jnp.sum(u[..., 1:4]**2, axis=-1)/rho
        min_rho = jnp.minimum(min_rho, rho.min())
        min_thermal = jnp.minimum(min_thermal, e.min())
    row = jnp.concatenate((jnp.reshape(dt, (1,)), box_means(U),
                           jnp.stack((work, loss, min_rho, min_thermal))))
    fields = {}
    if record:
        fields = step_record(initial_state_at_step, U, stages, dt, tau, gamma, args.replay_factor)
        fields["ou_coefficients"] = forcing[0][index]
        fields["ou_index"] = index
    return U, dt, index, raw, row, fields


cycle, dt, index = 0, jnp.asarray(np.inf, dtype=U.dtype), jnp.int32(-1)
scalar = np.dtype(U.dtype).type
raw = jnp.zeros_like(U[..., 1:4])
next_log = 0.5
starts = iter(sorted(args.replay_at))
next_window = next(starts, np.inf)
window_left = 0
if args.spectrum_dt:
    shell = mode_shells(initial.shape[:3])
    spectrum_times, spectra = [0.], [velocity_spectrum(initial, shell)]
    next_spectrum = args.spectrum_dt
wall = monotonic()
with history.open("x", buffering=65536) as stream:
    stream.write("cycle,time,dt,mass,mx,my,mz,total,kinetic_lab,thermal,drive_work,cooling_loss,"
                 "min_stage_rho,min_stage_thermal,wall_seconds,write_seconds,replay\n")
    initial_thermal = initial[..., 4] - 0.5*np.sum(initial[..., 1:4]**2, axis=-1)/initial[..., 0]
    first = [0, 0, 0, *np.asarray(box_means(U)), 0, 0, initial[..., 0].min(), initial_thermal.min(), 0, 0, 0]
    stream.write(",".join(format(x, ".17g") for x in first) + "\n")
    current = 0.0
    while current < tlim and cycle < time["nlim"]:
        if current >= next_window:
            if window_left:
                raise ValueError("requested replay windows overlap")
            window = out.parent / f"window_{cycle:06d}"
            window.mkdir()
            window_left, next_window = args.replay_steps, next(starts, np.inf)
        record = window_left > 0
        step_start = monotonic()
        remaining = scalar(tlim - current)
        U, dt, index, raw, diagnostics, fields = advance(
            U, scalar(current), dt, index, raw, forcing, remaining, record=record)
        row, fields = jax.device_get((diagnostics, fields))
        elapsed = monotonic() - step_start
        cycle += 1
        if not np.isfinite(row).all() or row[0] <= 0:
            raise RuntimeError(f"nonfinite data or stalled timestep at cycle {cycle}")
        if row[-2] < dfloor or row[-1] < pfloor/(gamma-1):
            raise RuntimeError(f"EOS floor reached at cycle {cycle}")
        # Keep the physical clock in double precision even when the GPU evolves float32.
        step_dt = tlim - current if row[0] == remaining else float(row[0])
        write_seconds = 0.
        if record:
            write_start = monotonic()
            path = window / f"step_{cycle:06d}.npz"
            tmp = path.with_suffix(".tmp")
            with tmp.open("xb") as stream_out:
                np.savez(stream_out, cycle=cycle, time=current, dt=step_dt, **fields)
                stream_out.flush()
                os.fsync(stream_out.fileno())
            tmp.replace(path)
            write_seconds = monotonic() - write_start
            window_left -= 1
        current = tlim if row[0] == remaining else current + step_dt
        if args.spectrum_dt and (current >= next_spectrum or current == tlim):
            spectrum_times.append(current)
            spectra.append(velocity_spectrum(U, shell))
            next_spectrum = (int(current/args.spectrum_dt) + 1)*args.spectrum_dt
        stream.write(",".join(format(x, ".17g") for x in
                              (cycle, current, step_dt, *row[1:], elapsed, write_seconds, int(record))) + "\n")
        if current >= next_log or current == tlim:
            stream.flush()
            print(f"t={current:.4f}  cycle={cycle}  K={row[6]:.6f}  "
                  f"thermal={row[7]:.6f}  elapsed={monotonic()-wall:.0f}s", flush=True)
            next_log += 0.5
if current != tlim:
    raise RuntimeError(f"cycle limit reached at t={current:g}")
if window_left or np.isfinite(next_window):
    raise RuntimeError("the run ended before all requested replay steps were saved")
if args.spectrum_dt:
    np.savez(spectrum_path, time=spectrum_times, mode=np.arange(len(spectra[0])),
             velocity_energy=spectra, requested_interval=args.spectrum_dt)
np.savez(out, **dict(zip("xyz", coords)), initial=initial, conserved=np.asarray(U),
         time=current, dt=float(dt), cycle=cycle, problem=problem["name"], sgs_model="",
         eos="ideal", gamma=gamma, settings=json.dumps(settings))
print(out.resolve(), flush=True)
