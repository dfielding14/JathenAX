from pathlib import Path
import json, shutil
import jax
import jax.numpy as jnp
import numpy as np
from jathenax.input import read_input
from jathenax.hydro import recover_primitives, hydro_step, hydro_timestep
from jathenax.forcing import update_force
from jathenax.run import prepare_forcing
from validation.replay_window import step_record

root = Path(__file__).resolve().parent
source = root.parent / "stationary_mach05_jathenax/dedt1_32_seed1618_tcorr044"
cfg = read_input(source / "input.toml")
shutil.copyfile(source / "input.toml", root / "input.toml")
with np.load(source / "final.npz") as f:
    u = jnp.asarray(f["conserved"])
    coords = tuple(f[axis] for axis in "xyz")
    time, previous_dt, cycle = float(f["time"]), float(f["dt"]), int(f["cycle"])
assert u.shape == (32,32,32,5) and u.dtype == np.float64
window = root / f"window_{cycle:06d}"
window.mkdir()
h = cfg["hydro"]
drive = cfg["turb_driving"]
tau = cfg["hydro_srcterms"]["tau_cool"]
gamma = h["gamma"]
start = time
limit = start + 0.5
spacing = tuple(1/len(x) for x in coords)
forcing = prepare_forcing(drive, coords, limit, u.dtype)

@jax.jit
def advance(u, t, last_dt, index, raw, forcing):
    _, w = recover_primitives(u, "ideal", gamma, h["dfloor"], h["pfloor"])
    dt = hydro_timestep(w, spacing, cfg["time"]["cfl_number"], last_dt, t, limit, gamma=gamma, tau_cool=tau)
    index, raw = update_force(t, index, raw, *forcing, drive["dt_update"], drive["tdriv_start"])
    end, stages = hydro_step(u, dt, spacing, gamma=gamma, tau_cool=tau, raw_force=raw, dedt=drive["dedt"],
                             dfloor=h["dfloor"], pfloor=h["pfloor"], integrator=cfg["time"]["integrator"],
                             reconstruct=h["reconstruct"], rsolver=h["rsolver"], trace=True)
    fields = step_record(u, end, stages, dt, tau, gamma, factor=4)
    fields.update(ou_coefficients=forcing[0][index], ou_index=index)
    return end, dt, index, raw, fields

# A forced refresh reconstructs the absolute update at t=40 from the original seed.
index, raw = jnp.int32(-1), jnp.zeros_like(u[...,1:4])
for _ in range(32):
    u, previous_dt, index, raw, fields = advance(u, np.float64(time), jnp.float64(previous_dt), index, raw, forcing)
    step_dt = float(previous_dt)
    cycle += 1
    with (window / f"step_{cycle:06d}.npz").open("xb") as out:
        np.savez(out, cycle=cycle, time=time, dt=step_dt, **jax.device_get(fields))
    time += step_dt
assert time < limit and np.isfinite(np.asarray(u)).all()
(root / "source.json").write_text(json.dumps(dict(snapshot=str((source/"final.npz").relative_to(root.parents[1])), first_time=start,
    last_time=time, seed=drive["rseed"], steps=32, backend=str(u.device), precision=str(u.dtype)),indent=2)+"\n")
print(f"Saved {window} at t={start:.6f} through t={time:.6f}", flush=True)
