"""Measure the box Mach number and calibrate linear cooling from an ideal-hydro history."""

import argparse
import json
from configparser import ConfigParser
from pathlib import Path

import numpy as np


def time_mean(time, values, left, right):
    """Integrate the sampled history up to the exact boundaries of a time block."""
    interior = time[(time > left) & (time < right)]
    t = np.r_[left, interior, right]
    y = np.interp(t, time, values)
    return np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(t)) / (right - left)


def correlation_time(samples, spacing):
    """Integrate the normalized autocorrelation through its first positive lobe."""
    centered = samples - samples.mean()
    corr = np.correlate(centered, centered, mode="full")[len(samples) - 1:]
    if corr[0] == 0:
        return 0.0
    stop = np.flatnonzero(corr[1:] <= 0)
    end = stop[0] + 1 if len(stop) else len(corr)
    return spacing * (0.5 + corr[1:end].sum() / corr[0])


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("deck", type=Path, help="Input deck with the parameters used for this run")
parser.add_argument("history", type=Path, help="Native per-step history.csv or AthenaK ideal-hydro .hst file")
parser.add_argument("--interval", type=float, nargs=2, metavar=("START", "END"),
                    help="Default is the second half of a native run or 60 to 120 for AthenaK")
parser.add_argument("--blocks", type=int, default=6, help="Number of equal-duration time blocks")
args = parser.parse_args()
target_mach = 0.5
native = args.deck.suffix == ".toml"
if native:
    from jathenax.input import read_input

    deck = read_input(args.deck)
    gamma, tau = deck["hydro"]["gamma"], deck["hydro_srcterms"]["tau_cool"]
    power, seed = deck["turb_driving"]["dedt"], deck["turb_driving"]["rseed"]
    grid = [deck["mesh"][k] for k in ("nx", "ny", "nz")]
    rows = np.genfromtxt(args.history, names=True, delimiter=",", ndmin=1)
    if not all(np.isfinite(rows[k]).all() for k in rows.dtype.names):
        parser.error("nonfinite native history")
    time, mass, total = rows["time"], rows["mass"], rows["total"]
    if (len(time) < 2 or time[0] != 0 or np.any(np.diff(time) <= 0)
            or not np.array_equal(rows["cycle"], np.arange(len(time)))
            or not np.allclose(np.diff(time), rows["dt"][1:], rtol=1e-11, atol=1e-14)):
        parser.error("expected consecutive native timesteps from t=0")
    volume = 1.0
    lab_kinetic = rows["kinetic_lab"]
    bulk_kinetic = sum(rows[k]**2 for k in ("mx", "my", "mz")) / (2 * mass)
else:
    deck = ConfigParser(interpolation=None, inline_comment_prefixes=("#",))
    deck.read_string(args.deck.read_text().replace("<", "[").replace(">", "]"))
    gamma = deck.getfloat("hydro", "gamma")
    tau = deck.getfloat("problem", "tau_cool")
    power = deck.getfloat("turb_driving", "dedt")
    seed = deck.getint("turb_driving", "rseed")
    mesh = deck["mesh"]
    grid = [mesh.getint(f"nx{i}") for i in (1, 2, 3)]
    volume = np.prod([mesh.getfloat(f"x{i}max") - mesh.getfloat(f"x{i}min") for i in (1, 2, 3)])
    with args.history.open() as stream:
        stream.readline()
        labels = [token.split("=", 1)[1] for token in stream.readline().split() if "=" in token]
    if labels[:10] != ["time", "dt", "mass", "1-mom", "2-mom", "3-mom", "tot-E", "1-KE", "2-KE", "3-KE"]:
        parser.error("expected AthenaK ideal-hydro energy and momentum columns")
    history = np.loadtxt(args.history, usecols=range(10), ndmin=2)
    if not np.isfinite(history).all() or np.any(np.diff(history[:, 0]) < 0):
        parser.error("nonfinite values or backward time in the history")
    # Finalization can repeat a timestamp with a different next dt and identical physical state.
    repeat = np.flatnonzero(np.diff(history[:, 0]) == 0)
    if not np.array_equal(history[repeat, 2:], history[repeat + 1, 2:]):
        parser.error("different physical states at the same history time")
    history = np.delete(history, repeat, axis=0)
    time, mass = history[:, 0], history[:, 2]
    total = history[:, 6] / volume
    lab_kinetic = history[:, 7:10].sum(axis=1) / volume
    bulk_kinetic = (history[:, 3:6]**2).sum(axis=1) / (2 * mass * volume)
if not (gamma > 1 and tau > 0 and power > 0 and volume > 0):
    parser.error("expected positive cooling time, driving power and volume, and gamma > 1")
start, end = args.interval or ((0.5*time[-1], time[-1]) if native else (60., 120.))
if not (time[0] <= start < end <= time[-1]) or args.blocks < 2:
    parser.error("select a recorded interval and at least two blocks")
interval_samples = time[(time >= start) & (time <= end)]
if len(interval_samples) < 2 * args.blocks or np.any(mass <= 0):
    parser.error("insufficient history sampling or nonpositive mass")

# Subtract COM motion only from kinetic energy. Thermal energy uses full lab-frame KE.
kinetic = lab_kinetic - bulk_kinetic
thermal = total - lab_kinetic
if np.any(thermal <= 0):
    parser.error("nonpositive thermal energy in the history")
sum_roundoff = 4 * np.prod(grid) * np.finfo(float).eps
if np.any(kinetic < -sum_roundoff * (lab_kinetic + bulk_kinetic)):
    parser.error("bulk kinetic energy exceeds total kinetic energy in the history")
kinetic = np.maximum(kinetic, 0)
factor = 2 / (gamma * (gamma - 1))
edges = np.linspace(start, end, args.blocks + 1)
block_k, block_u = np.array([[time_mean(time, values, left, right)
                              for left, right in zip(edges[:-1], edges[1:])]
                             for values in (kinetic, thermal)])
k_mean, u_mean = block_k.mean(), block_u.mean()
mach = np.sqrt(factor * k_mean / u_mean)
block_mach = np.sqrt(factor * block_k / block_u)
# Removing paired blocks propagates covariance of kinetic and thermal energy through their ratio.
leave_one_out = np.sqrt(factor * (block_k.sum() - block_k) / (block_u.sum() - block_u))
mach_se = np.sqrt((args.blocks - 1) * np.mean((leave_one_out - leave_one_out.mean())**2))

cadence = np.median(np.diff(interval_samples))
regular = np.linspace(start, end, round((end - start) / cadence) + 1)
energy_histories = {"kinetic": kinetic, "thermal": thermal}
acf_times = {name: correlation_time(np.interp(regular, time, values), regular[1] - regular[0])
             for name, values in energy_histories.items()}
storage_rate = (np.interp(end, time, total) - np.interp(start, time, total)) / (end - start)
cooling_ratio = u_mean / (tau * power)
if native and deck["hydro_srcterms"]["cooling_ramp_time"] > 0:
    loss_start, loss_end = np.interp([start, end], time, np.cumsum(rows["cooling_loss"]))
    cooling_ratio = (loss_end - loss_start) / ((end - start) * power)
half_intervals = ((start, 0.5 * (start + end)), (0.5 * (start + end), end))
report = {
    "seed": seed, "tau_cool": tau, "target_mach": target_mach,
    "gamma": gamma, "driving_power_per_volume": power,
    "grid_xyz": grid,
    "interval": [start, end], "block_duration": (end - start) / args.blocks,
    "mean_turbulent_kinetic_density": k_mean, "mean_thermal_density": u_mean,
    "mach": mach, "mach_block_standard_error": mach_se, "block_mach": block_mach.tolist(),
    "correlation_times": acf_times,
    "half_mean_energies": [
        {name: time_mean(time, values, left, right) for name, values in energy_histories.items()}
        for left, right in half_intervals
    ],
    "mean_cooling_over_power": cooling_ratio, "storage_rate_over_power": storage_rate / power,
    "history_budget_residual_over_power": storage_rate / power - (1 - cooling_ratio),
    "tau_from_measured_kinetic": factor * k_mean / (power * target_mach**2),
}
if native:
    driving = deck["turb_driving"]
    driving_length = 1.0 / driving["npeak"]  # The native periodic box has unit side length.
    rms_speed = np.sqrt(2 * k_mean / time_mean(time, mass, start, end))
    report["forcing_correlation_time"] = driving["tcorr"]
    report["peak_forcing_wavelength"] = driving_length
    report["turbulent_rms_speed"] = rms_speed
    report["eddy_turnover_time"] = driving_length / rms_speed if rms_speed > 0 else None
    report["forcing_correlation_over_turnover"] = driving["tcorr"] * rms_speed / driving_length
    # Restrict budget sums to whole steps even when a requested boundary falls inside one.
    first, last = np.searchsorted(time, start), np.searchsorted(time, end, side="right") - 1
    steps = slice(first + 1, last + 1)
    duration = time[last] - time[first]
    work, loss = rows["drive_work"][steps], rows["cooling_loss"][steps]
    change = total[last] - total[first]
    energy_steps = np.diff(total[first:last + 1])
    report["measured_energy_budget"] = {
        "time_start": time[first], "time_end": time[last], "steps": int(last - first),
        "driving_work": work.sum(), "cooling_loss": loss.sum(), "energy_change": change,
        "mean_driving_rate": work.sum()/duration, "mean_cooling_rate": loss.sum()/duration,
        "mean_storage_rate": change/duration, "cooling_over_drive": loss.sum()/work.sum(),
        "relative_closure_residual": (change - np.sum(work - loss)) / work.sum(),
        "max_abs_one_step_residual": np.max(np.abs(energy_steps - (work - loss))),
        "max_abs_injected_power_error": np.max(np.abs(work/rows["dt"][steps] - power)),
        "min_stage_density": rows["min_stage_rho"][steps].min(),
        "min_stage_thermal_density": rows["min_stage_thermal"][steps].min(),
    }
    report["block_mean_kinetic"] = block_k.tolist()
    report["block_mean_thermal"] = block_u.tolist()
print(json.dumps(report, indent=2, allow_nan=False))
