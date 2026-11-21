"""Close the cell-by-cell conservation budget in an AthenaK or native replay window."""

import argparse
import json
from pathlib import Path

import numpy as np


def check_roundoff(error, scale, tolerance, label):
    """Compare with roundoff in the quantities added to form this budget."""
    bad = np.abs(error) > tolerance * scale
    if np.any(bad):
        index = np.unravel_index(bad.argmax(), bad.shape)
        raise ValueError(f"{label} at index {tuple(map(int, index))}: "
                         f"residual {error[index]:.6g}, bound {tolerance * scale[index]:.6g}")


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("directory", type=Path, help="Directory containing window.CYCLE.bin or step_CYCLE.npz files")
args = parser.parse_args()
paths = sorted(args.directory.glob("window.*.bin"), key=lambda p: int(p.name.split(".")[1]))
native = sorted(args.directory.glob("step_*.npz"), key=lambda p: int(p.stem.split("_")[-1]))
if native:
    from jathenax.input import read_input

    if paths:
        parser.error("the window contains both native and AthenaK records")
    paths = native
    deck = read_input(args.directory.parent / "input.toml")
    fine_shape = tuple(deck["mesh"][k] for k in ("nx", "ny", "nz"))
    dedt, tau, gamma = deck["turb_driving"]["dedt"], deck["hydro_srcterms"]["tau_cool"], deck["hydro"]["gamma"]
if not paths:
    parser.error(f"no reference steps in {args.directory}")

fields = ("mass", "momentum_x", "momentum_y", "momentum_z", "total_energy")
cell_axes = (1, 2, 3)
eps = np.finfo(np.float64).eps
peak_step_error = np.zeros(5)
peak_partition_error = driving_work = cooling_loss = subgrid_driving_work = 0.0

for step, path in enumerate(paths):
    if native:
        with np.load(path) as rec:
            factor = int(rec.get("filter_factor", 2))  # Earlier native records always used 2:1.
            if factor < 1 or any(n % factor for n in fine_shape):
                raise ValueError(f"invalid filter factor in {path.name}")
            nx, ny, nz = (n // factor for n in fine_shape)
            dx, dy, dz = 1/nx, 1/ny, 1/nz
            cycle, time, dt = int(rec["cycle"]), float(rec["time"]), float(rec["dt"])
            states, final = rec["stage_state"], rec["final_state"]
            eps = np.finfo(final.dtype).eps
            if states.shape != (3, nz, ny, nx, 7) or cycle != int(path.stem.split("_")[-1]):
                raise ValueError(f"inconsistent native grid or cycle in {path.name}")
            stress = rec["stage_stress"]
            np.testing.assert_allclose(0.5*(stress[..., 0] + stress[..., 3] + stress[..., 5]),
                                       states[..., 6], rtol=16*eps, atol=0)
            for key in ("stage_state", "stage_stress", "stage_enthalpy_flux", "stage_q3",
                        "ou_coefficients", "force_scale", "force_mean"):
                if not np.isfinite(rec[key]).all():
                    raise ValueError(f"nonfinite {key} in {path.name}")
            pre, impulse, work = (rec[k].astype(np.float64) for k in ("pre_kick", "impulse", "work"))
            kick = np.concatenate((np.zeros_like(work[..., None]), impulse, work[..., None]), axis=-1)
            scale = np.abs(pre) + np.abs(final[..., :5]) + np.abs(kick)
            # Opposite fine momenta can cancel inside a coarse cell. Bound their unsigned
            # means by sqrt(2 rho_bar KE_bar); a kinetic kick preserves the fine thermal energy.
            for q in (pre, final.astype(np.float64)):
                ke = np.maximum(q[..., 4] - final[..., 5], 0.) + eps*np.abs(q[..., 4])
                scale[..., 1:4] += 2*np.sqrt(2*q[..., 0]*ke)[..., None]
            check_roundoff(final[..., :5] - pre - kick, scale, 32*eps, f"{path.name} kick")
            volume = dx*dy*dz
            faces = rec["face_flux_dt"].astype(np.float64)
            parts = [states[0], final, *(face*(volume/h) for face, h in zip(faces, (dx, dy, dz))),
                     impulse*volume, work[..., None]*volume, -rec["cooling_loss"][..., None]*volume]
            data = np.moveaxis(np.concatenate(parts, axis=-1), -1, 0)
    else:
        cycle = int(path.name.split(".")[1])
        with path.open("rb") as stream:
            header = stream.readline().split()
            nx, ny, nz, factor = map(int, header[:4])
            dx, dy, dz, time, dt, dedt, tau, gamma = map(float, header[4:])
            if path.stat().st_size - stream.tell() != 34 * nx * ny * nz * 8:
                raise ValueError(f"incomplete or oversized record in {path.name}")
            data = np.fromfile(stream, dtype=np.float64).reshape(34, nz, ny, nx)
    if (min(nx, ny, nz, factor) < 1 or min(dx, dy, dz, dt, tau) <= 0 or dedt < 0
            or gamma <= 1 or not np.isfinite([dx, dy, dz, time, dt, dedt, tau, gamma]).all()):
        raise ValueError(f"invalid grid, time or physics parameters in {path.name}")
    if not np.isfinite(data).all():
        raise ValueError(f"nonfinite record in {path.name}")
    initial, final = data[:7], data[7:14]
    settings = (nx, ny, nz, factor, dx, dy, dz, dedt, tau, gamma)
    if step == 0:
        first_cycle, first_time, first_state = cycle, time, initial[:5].copy()
        window_settings = settings
        window_update = np.zeros_like(first_state)
        window_scale = np.zeros_like(first_state)
        cooling_only = cycle == 0 and dedt == 0
        exact_thermal = 1 / (gamma * (gamma - 1))
        peak_cooling_error = 0.0
    else:
        if cycle != last_cycle + 1 or time != end_time or settings != window_settings:
            raise ValueError(f"discontinuous cycle, time, grid or physics parameters at {path.name}")
        if native:
            np.testing.assert_array_equal(initial[:5], last_state[:5], err_msg=f"discontinuous conserved state at {path.name}")
            # XLA may fuse the two evaluations of the same thermal and SGS block moments differently.
            check_roundoff(initial[5:] - last_state[5:], np.abs(initial[5:]) + np.abs(last_state[5:]),
                           (4 + factor**3)*eps, f"{path.name} derived-moment continuity")
        else:
            np.testing.assert_array_equal(initial, last_state, err_msg=f"discontinuous state at {path.name}")
    last_cycle, end_time, last_state = cycle, time + dt, final
    volume = dx * dy * dz
    tolerance = (16 + 2 * factor**3) * eps

    # Each stored face is the low face. Its neighbor supplies the outgoing flux.
    faces = data[14:29].reshape(3, 5, nz, ny, nx)
    impulse, work, cooling = data[29:32], data[32], data[33]
    integral = np.zeros_like(initial[:5])
    magnitude = np.zeros_like(integral)
    for direction, low in enumerate(faces):
        high = np.roll(low, -1, axis=3 - direction)
        integral += low - high
        magnitude += np.abs(low) + np.abs(high)
    integral[1:4] += impulse
    integral[4] += work + cooling
    magnitude[1:4] += np.abs(impulse)
    magnitude[4] += np.abs(work) + np.abs(cooling)
    update = integral / volume
    scale = np.abs(initial[:5]) + np.abs(final[:5]) + magnitude / volume
    residual = final[:5] - initial[:5] - update
    check_roundoff(residual, scale, tolerance, f"{path.name} cell budget (field,z,y,x)")
    peak_step_error = np.maximum(peak_step_error, np.abs(residual).max(axis=cell_axes))
    window_update += update
    window_scale += scale

    # Thermal energy and the fine kinetic moment were restricted independently.
    for state in (initial, final):
        density, momentum = state[0], state[1:4]
        energy, thermal, sgs = state[4:7]
        if np.any(density <= 0) or np.any(thermal <= 0):
            raise ValueError(f"nonpositive density or thermal energy in {path.name}")
        resolved = 0.5 * (momentum**2).sum(axis=0) / density
        if np.any(sgs < -tolerance * (resolved + np.abs(sgs))):
            raise ValueError(f"negative SGS kinetic energy in {path.name}")
        partition_error = energy - thermal - resolved - sgs
        check_roundoff(partition_error, np.abs(energy) + thermal + resolved + np.abs(sgs),
                       tolerance, f"{path.name} energy partition")
        peak_partition_error = max(peak_partition_error, np.abs(partition_error).max())

    # An unforced run from the specified uniform initial condition isolates the cooling operator.
    if cooling_only:
        h = dt / tau
        exact_thermal *= 1 - h + h**2 / 2 - h**3 / 6
        np.testing.assert_allclose(final[5], exact_thermal, rtol=tolerance, atol=0)
        np.testing.assert_array_equal(final[0], 1.0)
        np.testing.assert_array_equal(final[[1, 2, 3, 6]], 0.0)
        peak_cooling_error = max(peak_cooling_error, np.abs(final[5] - exact_thermal).max())

    # The edot driver removes net impulse and prescribes box-integrated work.
    if np.any(cooling > 0):
        raise ValueError(f"positive signed cooling source in {path.name}")
    box_tolerance = max(64*eps, tolerance) if native else max(tolerance, nx * ny * nz * factor**3 * eps)
    net_impulse = impulse.sum(axis=cell_axes)
    check_roundoff(net_impulse, np.abs(impulse).sum(axis=cell_axes), box_tolerance,
                   f"{path.name} net driving impulse")
    step_work = work.sum()
    target_work = dedt * nx * ny * nz * volume * dt
    check_roundoff(step_work - target_work, np.abs(work).sum() + target_work,
                   box_tolerance, f"{path.name} driving work")
    # The final momentum includes the kick, so subtract half the impulse for its midpoint.
    resolved_work = ((final[1:4] - 0.5 * impulse / volume) * impulse).sum(axis=0) / final[0]
    subgrid_driving_work += (work - resolved_work).sum()
    driving_work += step_work
    cooling_loss -= cooling.sum()

window_error = last_state[:5] - first_state - window_update
check_roundoff(window_error, window_scale, tolerance, "whole-window cell budget (field,z,y,x)")
report = {
    "first_cycle": first_cycle, "last_cycle": last_cycle, "recorded_steps": len(paths),
    "time_start": first_time, "time_end": end_time,
    "fine_cells_xyz": [factor * n for n in (nx, ny, nz)], "coarse_cells_xyz": [nx, ny, nz],
    "coarse_widths_xyz": [dx, dy, dz], "gamma": gamma,
    "driving_power_per_volume": dedt, "cooling_time": tau,
    "max_abs_cell_residual_per_step": dict(zip(fields, peak_step_error)),
    "max_abs_cell_residual_whole_window": dict(zip(fields, np.abs(window_error).max(axis=cell_axes))),
    "max_abs_energy_partition_residual": peak_partition_error,
    "box_driving_work": driving_work, "box_subgrid_driving_work": subgrid_driving_work,
    "box_cooling_loss": cooling_loss,
}
if cooling_only:
    report["max_abs_cooling_only_residual"] = peak_cooling_error
print(json.dumps(report, indent=2, allow_nan=False))
