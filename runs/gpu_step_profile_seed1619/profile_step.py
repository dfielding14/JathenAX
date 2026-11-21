"""Time the production pilot and matched-work variants at one saved developed state."""

import argparse
from contextlib import ExitStack, redirect_stdout
import csv
import io
import json
from pathlib import Path
import re
import runpy
import sys
import tempfile
from time import perf_counter
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from jathenax import hydro
from jathenax.forcing import render_force, update_force
from jathenax.input import read_input
from jathenax.run import prepare_forcing


def pilot_function(deck_path):
    # The production runner is a script. Let it run one warm-up step in a temporary
    # directory and use its actual per-step function without its buffer donation.
    with tempfile.TemporaryDirectory() as tmp:
        src = re.sub(r"(?m)^tlim\s*=.*$", "tlim = 0.00001", deck_path.read_text())
        src = re.sub(r"(?m)^file\s*=.*$", f'file = "{tmp}/final.npz"', src)
        deck = Path(tmp) / "input.toml"
        deck.write_text(src)
        with patch.object(sys, "argv", ["run_stationary_box", str(deck)]), redirect_stdout(io.StringIO()):
            runner = runpy.run_module("validation.run_stationary_box", run_name="__profile_pilot__")
    return runner["advance"].__wrapped__


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("directory", type=Path, help="completed pilot with input.toml, final.npz and history.csv")
parser.add_argument("--rounds", type=int, default=16)
parser.add_argument("--repeats", type=int, default=2)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--candidates", action="store_true", help="time benchmark-only rewrites in the full pilot")
parser.add_argument("--trajectory", type=int, default=0, help="check candidate timings over consecutive adaptive steps")
args = parser.parse_args()
directory, output = args.directory, args.output
output.mkdir(parents=True, exist_ok=True)
assert args.rounds >= 3 and args.repeats >= 1
deck_path = directory / "input.toml"
d = read_input(deck_path)
g, src, driv = d["hydro"], d["hydro_srcterms"], d["turb_driving"]
assert (d["time"]["integrator"], g["reconstruct"], g["rsolver"], g["eos"]) == ("rk3", "ppm4", "roe", "ideal")
assert not g["viscosity"] and not g["smagorinsky_cs"] and not src["drag_rate"] and not d["sgs"]["model"]
backend = jax.default_backend().lower()
jax.config.update("jax_enable_x64", backend != "metal")
dtype = np.float32 if backend == "metal" else np.float64
advance = pilot_function(deck_path)
with np.load(directory / "final.npz", allow_pickle=False) as f:
    u = jnp.asarray(f["conserved"], dtype=dtype)
    coords = tuple(f[key] for key in "xyz")
    start = float(f["time"])
n = len(coords[0])
spacing = tuple(1/len(a) for a in coords)
gamma, dfloor, pfloor = (g[k] for k in ("gamma","dfloor","pfloor"))
tau = src["tau_cool"]
physics = dict(gamma=gamma, dfloor=dfloor, pfloor=pfloor, tau_cool=tau,
               cooling_ramp_time=src["cooling_ramp_time"], cooling_ramp_exponent=src["cooling_ramp_exponent"],
               reconstruct=g["reconstruct"], rsolver=g["rsolver"], dedt=driv["dedt"])
history = np.genfromtxt(directory / "history.csv", names=True, delimiter=",")
wall = history["wall_seconds"][1:]
clock = history["time"][:-1].astype(dtype)
indices = np.floor((clock - dtype(driv["tdriv_start"])) / dtype(driv["dt_update"])).astype(int)
refreshed = np.diff(indices, prepend=-1) > 0
late = clock >= .5*start
frequency = float(refreshed[late].mean())
t = dtype(start)
previous = dtype(history["dt"][-2])  # Before the endpoint-clipped timestep.
remaining = dtype(1.)
ahead = 1. if args.trajectory else 2*driv["dt_update"]
forcing = prepare_forcing(driv, coords, start + ahead, u.dtype)
index = int(np.floor((t-dtype(driv["tdriv_start"]))/dtype(driv["dt_update"])))
held, refresh = jnp.int32(index), jnp.int32(index-1)
raw = jax.jit(render_force)(forcing[0][index], forcing[1])
base_args = (u, t, previous, held, raw, forcing, remaining)
new_args = (u, t, previous, refresh, raw, forcing, remaining)


def pilot(*inputs):
    return advance(*inputs, record=False)


def ordinary(u, t, previous, index, raw, forcing, remaining):
    _, w = hydro.recover_primitives(u, "ideal", gamma, dfloor, pfloor)
    dt = hydro.hydro_timestep(w, spacing, d["time"]["cfl_number"], previous, 0., remaining,
                              gamma=gamma, tau_cool=tau)
    index, raw = update_force(t, index, raw, *forcing, driv["dt_update"], driv["tdriv_start"])
    return hydro.hydro_step(u, dt, spacing, time=t, raw_force=raw, **physics), dt, index, raw


def core(u, dt, raw, t):
    return hydro.hydro_step(u, dt, spacing, time=t, raw_force=raw, **physics)


def capture(u, dt, raw, t):
    _, stages = hydro.hydro_step(u, dt, spacing, time=t, raw_force=raw, trace=True, **physics)
    faces = tuple((s[f"left_{k}"], s[f"right_{k}"]) for s in stages for k in "xyz")
    fluxes = tuple(s[f"flux_{k}"] for s in stages for k in "xyz")
    divs = tuple(sum((jnp.roll(s[f"flux_{k}"], -1, axis=2-i) - s[f"flux_{k}"]) / spacing[i]
                     for i, k in enumerate("xyz")) for s in stages)
    kick = tuple(stages[-1][k] for k in ("acceleration", "force_scale", "force_mean"))
    return faces, fluxes, divs, kick


def cut(kind):
    # Patches act only while JAX traces. Replay the original state at an RK-stage
    # boundary, keeping downstream numeric work and the returned 3D state live.
    def compute(u, dt, raw, t, cached, kick):
        values = iter(cached)
        rk = hydro.rk_stage
        def cached_div(state, saved, div, step_dt, coeff):
            if getattr(state, "ndim", 0):
                div = next(values)
            return rk(state, saved, div, step_dt, coeff)
        with ExitStack() as stack:
            if kind == "faces":
                stack.enter_context(patch.object(hydro, "reconstruct_faces", lambda *a, **kw: next(values)))
            elif kind == "fluxes":
                stack.enter_context(patch.object(hydro, "hydro_flux", lambda *a, **kw: next(values)))
            else:
                stack.enter_context(patch.object(hydro, "rk_stage", cached_div))
                if kind == "div_and_acceleration":
                    stack.enter_context(patch.object(hydro, "normalize_force", lambda *a, **kw: kick))
            return core(u, dt, raw, t)
    return compute


# Keep benchmark inputs on the device. Compilation and warm-up precede all timing.
def ready_call(fn, inputs):
    exe = jax.jit(fn).lower(*inputs).compile()
    def call():
        result = exe(*inputs)
        for leaf in jax.tree_util.tree_leaves(result):
            leaf.block_until_ready()
        return result
    return call


full_call = ready_call(pilot, base_args)
ref, dt, _, ref_raw, ref_row, _ = full_call()
core_args = (u, dt, raw, t)
calls = {"pilot_held": full_call, "pilot_refresh": ready_call(pilot, new_args)}
prefix = f"n{n}"
if args.candidates or args.trajectory:
    from jathenax import fluxes as flux_module, forcing as forcing_module
    from candidates import ppm_shared_edges, render_matmul, scalar_roe, yz_basis
    prefix += "_trials"
    yz = jax.jit(yz_basis)(forcing[1])
    def candidate(name):
        def compute(*all_inputs):
            *inputs, transverse = all_inputs
            with ExitStack() as stack:
                if name in ("ppm", "combined"):
                    stack.enter_context(patch.dict(flux_module.RECONSTRUCTIONS, ppm4=ppm_shared_edges))
                if name in ("roe", "combined"):
                    stack.enter_context(patch.object(hydro, "hydro_flux", scalar_roe))
                if name in ("render", "combined"):
                    stack.enter_context(patch.object(forcing_module, "render_force",
                                                    lambda a, b: render_matmul(a, b, transverse)))
                return pilot(*inputs)
        return compute
    if args.trajectory:
        assert args.trajectory > 1
        selected = ("pilot", "ppm", "render", "combined")
        executables = {name: jax.jit(candidate(name)).lower(*base_args, yz).compile() for name in selected}
        last_dt, last_index = dtype(history["dt"][-1]), jnp.int32(indices[-1])
        def trajectory(exe):
            state, current, prev, idx, field = u, start, last_dt, last_index, raw
            for _ in range(args.trajectory):
                state, prev, idx, field, row, _ = exe(state, dtype(current), prev, idx, field,
                                                     forcing, dtype(start + ahead - current), yz)
                values = np.asarray(row)
                assert np.isfinite(values).all() and values[-2] > dfloor and values[-1] > pfloor/(gamma-1)
                current += float(values[0])
            state.block_until_ready()
            return state, current
        reference, final_time = trajectory(executables["pilot"])
        ref_host = np.asarray(reference)
        del reference
        errors = {}
        for name, exe in executables.items():
            got, end = trajectory(exe)
            out = np.asarray(got)
            diff = out.astype(np.float64) - ref_host
            bits = np.uint32 if dtype == np.float32 else np.uint64
            errors[name] = dict(max_abs=float(np.abs(diff).max()),
                                relative_rms=float(np.linalg.norm(diff.ravel()) / np.linalg.norm(ref_host.ravel())),
                                bitwise_equal=bool(np.array_equal(out.view(bits), ref_host.view(bits))),
                                end_time_difference=end-final_time)
            assert errors[name]["relative_rms"] < 5e-5 and abs(end-final_time) < 1e-6, (name,errors[name])
        rows = []
        rng = np.random.default_rng(1619)
        for batch in range(args.rounds):
            row = {}
            for name in rng.permutation(selected):
                before = perf_counter()
                trajectory(executables[name])
                row[name] = (perf_counter()-before)*1000 / args.trajectory
            rows.append(row)
            print(f"{n}³ completed trajectory batch {batch+1}/{args.rounds}", flush=True)
        with (output/f"n{n}_trajectory_blocks.csv").open("w",newline="") as stream:
            writer = csv.DictWriter(stream, selected)
            writer.writeheader()
            writer.writerows(rows)
        report = dict(N=n, rounds=args.rounds, steps=args.trajectory, start=start, end=final_time,
                      result_errors=errors, ms_per_step={k: float(np.mean([r[k] for r in rows])) for k in selected})
        (output/f"n{n}_trajectory_meta.json").write_text(json.dumps(report, indent=2)+"\n")
        print(report, flush=True)
        sys.exit(0)
    for name in ("ppm", "roe", "render", "combined"):
        func = candidate(name)
        for branch, inputs in (("held", base_args), ("refresh", new_args)):
            calls[f"{name}_{branch}"] = ready_call(func, (*inputs, yz))
else:
    faces, fluxes, divs, kick = jax.block_until_ready(jax.jit(capture)(*core_args))
    calls.update(ordinary_held=ready_call(ordinary, base_args), core=ready_call(core, core_args))
    for kind, cache in (("faces", faces), ("fluxes", fluxes), ("div", divs), ("div_and_acceleration", divs)):
        calls[kind] = ready_call(cut(kind), (*core_args, cache, kick))
errors = {}
raw_errors = {}
for key, call in calls.items():
    out = call()
    state = out[0] if isinstance(out, tuple) else out
    err = float(jnp.max(jnp.abs(state-ref)))
    errors[key] = err
    assert err <= (2e-5 if dtype == np.float32 else 5e-12), (key, err)
    if args.candidates:
        rms_error = float(jnp.sqrt(jnp.mean((out[3]-ref_raw)**2) / jnp.mean(ref_raw**2)))
        raw_errors[key] = rms_error
        assert rms_error < (5e-6 if dtype == np.float32 else 2e-12), (key, rms_error)
        np.testing.assert_allclose(np.asarray(out[4]), np.asarray(ref_row), rtol=2e-5, atol=2e-6)
    call()
print(f"{n}³ on {u.device}, {u.dtype}, refresh fraction {frequency:.3f}, max state error {max(errors.values()):g}", flush=True)

rng = np.random.default_rng(1619)
samples = []
for batch in range(args.rounds):
    row = {}
    for key in rng.permutation(list(calls)):
        before = perf_counter()
        for _ in range(args.repeats):
            calls[key]()
        row[str(key)] = 1000*(perf_counter()-before) / args.repeats
    samples.append(row)
    if (batch+1)%4 == 0:
        print(f"{n}³ finished timing batch {batch+1}/{args.rounds}", flush=True)
with (output/f"{prefix}_blocks.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, list(calls))
    writer.writeheader()
    writer.writerows(samples)
held_rows = late & ~refreshed
new_rows = late & refreshed
archive = dict(late_mean_ms=float(1000*wall[late].mean()), refresh_fraction=frequency,
               late_steps=int(late.sum()), late_refresh_ms=float(1000*wall[new_rows].mean()))
if held_rows.any():
    archive["late_held_ms"] = float(1000*wall[held_rows].mean())
    matched = np.interp(clock[new_rows], clock[held_rows], wall[held_rows])
    archive["locally_paired_render_fraction"] = float((wall[new_rows]-matched).sum()/wall[late].sum())
meta = dict(N=n, cpu_backend=backend=="cpu", backend=backend, dtype=str(u.dtype), device=str(u.device),
            jax=jax.__version__, rounds=args.rounds, repeats=args.repeats, start=start, dt=float(dt),
            forcing_modes=int(forcing[0].shape[1]), archive=archive, max_abs_state_errors=errors,
            relative_rms_force_errors=raw_errors)
(output/f"{prefix}_meta.json").write_text(json.dumps(meta, indent=2)+"\n")
print(archive, {key: round(np.mean([r[key] for r in samples]),3) for key in calls}, flush=True)
