"""Feed the same filtered RK3 records from disk or a running fine box."""

from pathlib import Path
from time import monotonic

import jax
import jax.numpy as jnp
import numpy as np

from jathenax.forcing import update_force
from jathenax.hydro import hydro_step, hydro_timestep, recover_primitives
from jathenax.les import GAMMA, kinetic_work
from jathenax.run import prepare_forcing
from jathenax.tbnn import deviatoric
from validation.replay_window import step_record


def sample(record):
    impulse = jnp.asarray(record["impulse"])
    return dict(stage=jnp.asarray(record["stage_state"]), final=jnp.asarray(record["final_state"]),
                stress=deviatoric(jnp.asarray(record["stage_stress"])),
                heat=jnp.asarray(record["stage_enthalpy_flux"]), third=jnp.asarray(record["stage_q3"]),
                impulse=impulse, sgs_work=record["work"] - kinetic_work(record["pre_kick"], impulse))


def block(records, times, dts, endpoints):
    """Assign each scored endpoint the physical time since the preceding comparison."""
    dts, times = np.asarray(dts, dtype=float), np.asarray(times, dtype=float)
    ends = np.asarray(endpoints)
    if (not len(ends) or ends[-1] != len(dts)-1 or ends[0] < 0 or np.any(np.diff(ends) <= 0)
            or not np.isfinite(dts).all() or np.any(dts <= 0) or not np.isfinite(times).all()
            or not np.allclose(times[1:], (times+dts)[:-1], rtol=0, atol=1e-12)):
        raise ValueError("reference times must be contiguous with increasing comparison endpoints")
    data = jax.tree_util.tree_map(lambda *a: jnp.stack(a), *records)
    durations = np.zeros_like(dts)
    durations[ends] = np.diff(np.r_[0., np.cumsum(dts)[ends]])
    dtype = data["stage"].dtype
    data.update(dt=jnp.asarray(dts, dtype), duration=jnp.asarray(durations, dtype))
    return data, (times, dts, ends)


def read_saved(directory):
    records, times, dts, cycles = [], [], [], []
    for path in sorted(Path(directory).glob("step_*.npz")):
        with np.load(path, allow_pickle=False) as f:
            if f["stage_state"].shape != (3, 8, 8, 8, 7) or int(f["filter_factor"]) != 4:
                raise ValueError(f"unexpected 32-to-8 reference in {path}")
            records.append(sample(f))
            times.append(float(f["time"]))
            dts.append(float(f["dt"]))
            cycles.append(int(f["cycle"]))
    if len(records) != 32 or np.any(np.diff(cycles) != 1):
        raise ValueError("the first pilot requires 32 consecutive fine steps")
    return block(records, times, dts, np.arange(3, 32, 4))


def training_batch(data, comparisons):
    """Reuse compiled shapes for adaptive fine steps between the same number of targets."""
    extra = -len(data["dt"]) % comparisons
    if extra == 0:
        return data
    batch = {}
    for key, field in data.items():
        if key in ("dt", "duration", "impulse", "sgs_work"):
            tail = jnp.zeros((extra, *field.shape[1:]), field.dtype)
        else:
            tail = jnp.broadcast_to(field[-1], (extra, *field.shape[1:]))
        batch[key] = jnp.concatenate((field, tail))
    return batch


def save_block(path, data, clock):
    """Retain round-time targets, sampled stress labels and every kick for frozen replay."""
    times, dt, ends = clock
    first = np.r_[0, ends[:-1]+1]
    with path.open("xb") as stream:
        np.savez_compressed(stream, time=times, dt=dt, endpoints=ends, target_time=(times+dt)[ends],
                            initial=data["stage"][0, 0], target=data["final"][ends],
                            moment_state=data["stage"][first, 0], moment_stress=data["stress"][first, 0],
                            impulse=data["impulse"], sgs_work=data["sgs_work"],
                            drive_work=data["drive_work"], cooling_loss=data["cooling_loss"])


def live_blocks(snapshot, cfg, clock, count, cadence, *, compile_times=None):
    """Advance to regular targets, optionally replaying an archived trajectory first."""
    with np.load(snapshot, allow_pickle=False) as f:
        u = jnp.asarray(f["conserved"])
        coords = tuple(f[k] for k in "xyz")
        current, last_dt = float(f["time"]), float(f["dt"])
    n = u.shape[0]
    if u.shape != (n, n, n, 5) or n % 4 or any(v.shape != (n,) for v in coords):
        raise ValueError("expected a cubic fine checkpoint compatible with the 4:1 filter")
    h, drive, src = cfg["hydro"], cfg["turb_driving"], cfg["hydro_srcterms"]
    if (h["eos"] != "ideal" or not np.isclose(h["gamma"], GAMMA) or h["viscosity"] != 0
            or h["smagorinsky_cs"] != 0 or src["drag_rate"] != 0 or current < src.get("cooling_ramp_time", 0.)
            or cfg["time"]["integrator"] != "rk3" or (h["reconstruct"], h["rsolver"]) != ("ppm4", "roe")):
        raise ValueError("expected inviscid ideal-gas PPM4/Roe/RK3 with cooling at full strength")
    targets = []
    end, comparisons = current, 8
    if clock is not None:
        times, original_dts, ends = clock
        if current != times[0] or n != 32:
            raise ValueError("fine checkpoint must match the archived initial time and grid")
        targets.append((times + original_dts)[ends])
        end, comparisons = targets[0][-1], len(ends)
    first_tick = int(np.floor(end / cadence)) + 1
    if first_tick * cadence <= end:
        first_tick += 1
    ticks = np.arange(first_tick, first_tick + count * comparisons).reshape(count, comparisons)
    targets.extend(ticks * cadence)
    limit = targets[-1][-1] + drive["dt_update"]
    forcing = prepare_forcing(drive, coords, limit, u.dtype)
    spacing, tau = (1/n,)*3, src["tau_cool"]

    @jax.jit
    def fine_dt(state, previous):
        _, w = recover_primitives(state, "ideal", GAMMA, h["dfloor"], h["pfloor"])
        return hydro_timestep(w, spacing, cfg["time"]["cfl_number"], previous, 0., jnp.inf,
                              gamma=GAMMA, tau_cool=tau)

    @jax.jit
    def advance(state, time, dt, index, raw, forcing):
        index, raw = update_force(time, index, raw, *forcing, drive["dt_update"], drive["tdriv_start"])
        final, stages = hydro_step(state, dt, spacing, gamma=GAMMA, dfloor=h["dfloor"],
                                   pfloor=h["pfloor"], tau_cool=tau, raw_force=raw, dedt=drive["dedt"],
                                   trace=True)
        record = step_record(state, final, stages, dt, tau, GAMMA, 4)
        data = sample(record)
        if clock is None:
            data.update(drive_work=record["work"].mean(), cooling_loss=record["cooling_loss"].mean())
        return final, index, raw, data

    index, raw = jnp.int32(-1), jnp.zeros_like(u[..., 1:4])
    if compile_times is not None:
        jax.block_until_ready((u, raw, forcing))
        start = monotonic()
        fine_dt = fine_dt.lower(u, u.dtype.type(last_dt)).compile()
        advance = advance.lower(u, u.dtype.type(current), u.dtype.type(last_dt), index, raw, forcing).compile()
        compile_times["fine_compile_seconds"] = monotonic() - start
    for number, comparison_times in enumerate(targets):
        records, starts, steps, endpoints = [], [], [], []
        for target in comparison_times:
            while current < target:
                safe = float(fine_dt(u, u.dtype.type(last_dt)))
                if not np.isfinite(safe) or safe <= 0:
                    raise FloatingPointError("invalid fine CFL step")
                remaining = target - current
                if clock is not None and number == 0:
                    dt = original_dts[len(steps)]
                    if dt > safe * (1 + 1e-5):
                        raise ValueError("archived fine step exceeds the live CFL limit")
                else:
                    dt = remaining / np.ceil(remaining / safe)
                u, index, raw, record = advance(u, u.dtype.type(current), u.dtype.type(dt),
                                                index, raw, forcing)
                records.append(record)
                starts.append(current)
                steps.append(dt)
                at_target = np.isclose(dt, remaining, rtol=0, atol=2e-14)
                current = float(target) if at_target else current + dt
                last_dt = dt
            endpoints.append(len(steps)-1)
        yield block(records, starts, steps, endpoints)
