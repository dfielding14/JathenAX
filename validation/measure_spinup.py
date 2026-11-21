"""Measure velocity-spectrum and energy-balance settling in the paired 32³ pilot."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from jathenax.input import read_input
from jathenax.spectra import mode_shells, shell_energy


def block_means(t, y, edges):
    """Average a piecewise-linear time series over exact physical-time intervals."""
    if edges[0] < t[0] or edges[-1] > t[-1] or np.any(np.diff(t) <= 0):
        raise ValueError("the spectrum times must cover every interval")
    boundary = np.column_stack([np.interp(edges, t, column) for column in y.T])
    means = []
    for i, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        inside = (t > left) & (t < right)
        x = np.r_[left, t[inside], right]
        z = np.vstack([boundary[i], y[inside], boundary[i+1]])
        means.append(np.sum((z[:-1]+z[1:])*np.diff(x)[:, None]/2, axis=0)/(right-left))
    return np.array(means)


def spectral_distance(spectrum, reference):
    return np.sqrt(np.mean(np.log(spectrum/reference)**2, axis=-1))


def settling(passes, count, width):
    """Mark the first sustained block sequence and count later turbulent excursions."""
    streaks = np.flatnonzero(np.convolve(passes.astype(int), np.ones(count, int), "valid") == count)
    failures = np.flatnonzero(~passes)
    result = dict(total_blocks=len(passes), passed_blocks=int(passes.sum()),
                  last_failing_block_start=float(failures[-1]*width) if len(failures) else None,
                  first_passing_block_start=None, confirmed_by=None)
    if len(streaks):
        i = int(streaks[0])
        result.update(first_passing_block_start=i*width, confirmed_by=(i+count)*width,
                      failures_after_confirmation=int(np.count_nonzero(~passes[i+count:])),
                      blocks_after_confirmation=len(passes)-i-count)
    return result


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("directory", type=Path, help="contains protocol.json, warm/, and control/")
root = parser.parse_args().directory
protocol = json.loads((root/"protocol.json").read_text())
te = protocol["turnover_time"]
width = protocol["block_turnovers"]
period = width*te
count = protocol["consecutive_passing_blocks"]
tol = protocol["cooling_tolerance"]
percentile = protocol["reference_percentile"]
lo, hi = protocol["shells"]
modes = np.arange(lo, hi+1)

runs = {}
for name in ("control", "warm"):
    folder = root/name
    deck = read_input(folder/"input.toml")
    history = np.genfromtxt(folder/"history.csv", names=True, delimiter=",")
    with np.load(folder/"velocity_spectra.npz", allow_pickle=False) as saved:
        times = saved["time"]
        np.testing.assert_array_equal(saved["mode"][modes], modes)
        spectra = saved["velocity_energy"][:, modes]
        np.testing.assert_allclose(saved["requested_interval"], protocol["spectrum_dt"])
    t = history["time"]
    work, cooling = (np.cumsum(history[k]) for k in ("drive_work", "cooling_loss"))
    np.testing.assert_allclose([times[0], times[-1]], [0., deck["time"]["tlim"]])
    np.testing.assert_allclose(np.diff(t), history["dt"][1:], rtol=1e-10)
    if np.max(np.diff(times)) > protocol["spectrum_dt"] + np.max(history["dt"]) + 1e-12:
        raise ValueError("missing a scheduled spectrum")
    edges = period*np.arange(int(t[-1]/period)+1)
    block_spectra = block_means(times, spectra, edges)
    # A boundary inside a fine step receives the corresponding fraction of its measured work or loss.
    loss_blocks = np.diff(np.interp(edges, t, cooling))
    work_blocks = np.diff(np.interp(edges, t, work))
    ratio = loss_blocks/work_blocks
    source = history["drive_work"][1:]-history["cooling_loss"][1:]
    change = np.diff(history["total"])
    residual = (change.sum()-source.sum())/work[-1]
    np.testing.assert_allclose(residual, 0., atol=2e-11)
    np.testing.assert_allclose(history["drive_work"][1:]/history["dt"][1:], deck["turb_driving"]["dedt"], rtol=1e-10)
    rho_mean = history["mass"]
    kinetic = history["kinetic_lab"]-sum(history[k]**2 for k in ("mx", "my", "mz"))/(2*rho_mean)
    gamma = deck["hydro"]["gamma"]
    speed2 = np.column_stack((2*kinetic/rho_mean, gamma*(gamma-1)*history["thermal"]/rho_mean))
    initial_speeds = np.sqrt(speed2[0])
    with np.load(folder/"final.npz", allow_pickle=False) as f:
        for key, ix in (("initial", 0), ("conserved", -1)):
            u = f[key]
            v = u[..., 1:4]/u[..., :1]
            v -= v.mean(axis=(0,1,2))
            spec = shell_energy(np.moveaxis(v,-1,0), mode_shells(u.shape[:3]))
            np.testing.assert_allclose(spectra[ix], spec[modes], rtol=1e-12, atol=1e-31)
            if key == "initial":
                total = spec.sum()
                outside = float(spec[4:].sum()/total) if total else None
                if total:
                    k = np.meshgrid(*(np.fft.fftfreq(n)*n for n in v.shape[:3]), indexing="ij", sparse=True)
                    vhat = np.fft.fftn(v, axes=(0,1,2))/v[...,0].size
                    k2 = sum(ki*ki for ki in k)
                    kv = sum(ki*vhat[..., i] for i, ki in enumerate(k[::-1]))
                    compressive = float((np.abs(kv)**2 / np.where(k2 > 0, k2, 1)).sum()/(2*total))
                else:
                    compressive = None
    runs[name] = dict(times=times, spectra=spectra, block_spectra=block_spectra, edges=edges,
                      ratio=ratio, history=history, deck=deck)
    runs[name]["report"] = dict(steps=int(history["cycle"][-1]), spectral_samples=len(times),
        initial_vrms=float(initial_speeds[0]), initial_cs_rms=float(initial_speeds[1]),
        initial_velocity_power_fraction_above_k3=outside, initial_compressive_power_fraction=compressive,
        max_abs_discrete_step_budget_residual=float(np.max(np.abs(change-source))),
        whole_run_budget_residual_over_injected_work=float(residual),
        late_mass_weighted_rms_speeds=np.sqrt(block_means(t,speed2,protocol["reference_interval"])[0]).tolist(),
        max_abs_one_turnover_imbalance=float(np.max(np.abs(ratio-1))),
        late_cooling_over_input=float(np.diff(np.interp(protocol["reference_interval"],t,cooling))[0] /
                                      np.diff(np.interp(protocol["reference_interval"],t,work))[0]),
        min_stage_rho=float(history["min_stage_rho"].min()),
        min_stage_thermal=float(history["min_stage_thermal"].min()))

control, warm = runs["control"], runs["warm"]
for key in ("mesh", "time", "hydro", "hydro_srcterms", "turb_driving"):
    assert control["deck"][key] == warm["deck"][key], key
np.testing.assert_allclose([warm["report"]["initial_vrms"], warm["report"]["initial_cs_rms"]], [1.14,2.28], atol=1e-12)
assert warm["report"]["initial_velocity_power_fraction_above_k3"] < 1e-20
assert warm["report"]["initial_compressive_power_fraction"] < 1e-20

# Establish a fluctuation threshold from the developed portion of the original setup.
left, right = protocol["reference_interval"]
ref_edges = left + np.arange(int((right-left)/period)+1)*period
ref_blocks = block_means(control["times"], control["spectra"], ref_edges)
reference = ref_blocks.mean(axis=0)
loo = (ref_blocks.sum(axis=0) - ref_blocks)/(len(ref_blocks)-1)
ref_dist = spectral_distance(ref_blocks, loo)
limit = float(np.percentile(ref_dist, percentile))
report = dict(turnover_time=te, complete_reference_blocks=len(ref_blocks),
              reference_block_extent=[float(ref_edges[0]), float(ref_edges[-1])],
              reference_percentile=percentile, rms_log_spectrum_threshold=limit,
              stationary_reference_cooling_tolerance=tol, reference_velocity_spectrum=reference.tolist())
for name, run in runs.items():
    s = run["report"]
    distance = spectral_distance(run["block_spectra"], reference)
    run["distance"] = distance
    spectral_ok = distance <= limit
    energy_ok = np.abs(run["ratio"]-1) <= tol
    s["spectrum"] = settling(spectral_ok, count, width)
    s["energy"] = settling(energy_ok, count, width)
    s["joint"] = settling(spectral_ok & energy_ok, count, width)
    s["energy_tolerance_sensitivity"] = {str(p):settling(np.abs(run["ratio"]-1) <= p, count, width)
                                          for p in (.01, .05)}
    s["spectrum_percentile_sensitivity"] = {str(p):settling(distance <= np.percentile(ref_dist, p), count, width)
                                             for p in (90, 99)}
    s["late_spectrum_distance"] = float(spectral_distance(
        block_means(run["times"],run["spectra"],[left,right])[0], reference))
    s["late_half_spectral_distances"] = spectral_distance(
        block_means(run["times"],run["spectra"],[left,(left+right)/2,right]), reference).tolist()
    np.savetxt(root/name/"settling_blocks.csv",
               np.column_stack((run["edges"][:-1]/te, run["edges"][1:]/te, distance,
                                run["ratio"], spectral_ok, energy_ok)),
               delimiter=",", fmt="%.10g", comments="",
               header="start_turnovers,end_turnovers,rms_log_spectrum_error,cooling_over_injection,spectrum_pass,energy_pass")
    report[name] = s
(root/"settling.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")

# Show the cascade forming and the individual one-turnover measurements behind the settling times.
fig = plt.figure(figsize=(10.6,5.4), layout="constrained")
gs = fig.add_gridspec(2,2,width_ratios=(1,1.35))
spec_ax = fig.add_subplot(gs[:,0])
dist_ax = fig.add_subplot(gs[0,1])
balance_ax = fig.add_subplot(gs[1,1], sharex=dist_ax)
spec_ax.fill_between(modes, *np.percentile(ref_blocks,[5,95],axis=0), color="#cddcda",
                     label="Stationary control, 5–95%")
spec_ax.loglog(modes, reference, color="#263945", lw=1.8, label="Stationary mean")
initial = warm["spectra"][0]
powered = initial > initial.max()*1e-12
spec_ax.loglog(modes[powered], initial[powered], "o:", ms=4, color="#a95a29", label="Initial velocity")
for i, color in ((0,"#bb7240"),(2,"#047388")):
    spec_ax.loglog(modes,warm["block_spectra"][i],".-",color=color,lw=1.4,
                   label=f"Turnovers {i}–{i+1}")
spec_ax.set(xticks=[1,2,3,5,8,12,15], xlabel="Box wavenumber $k$",
            ylabel=r"Shell velocity power $E_v(k)$  (code speed$^2$)")
spec_ax.set_xticklabels(["1","2","3","5","8","12","15"])
spec_ax.legend(loc="lower left",fontsize=8,frameon=False)
for name, color, ls, label in (("control", "#97816a", "--", "Previous start from rest"),
                               ("warm", "#08687a", "-", "Turbulent start")):
    run = runs[name]
    x = .5*(run["edges"][:-1]+run["edges"][1:])/te
    dist_ax.plot(x,run["distance"],color=color,ls=ls,lw=1.3,label=label)
    balance_ax.plot(x,100*(run["ratio"]-1),color=color,ls=ls,lw=1.3)
dist_ax.axhline(limit, color="#283b46",ls=":",lw=1.3,label="Stationary 95th percentile")
dist_ax.set(yscale="log",ylabel="RMS log spectral error")
dist_ax.legend(frameon=False,fontsize=8,loc="upper right")
balance_ax.axhspan(-100*tol,100*tol,color="#deede4")
balance_ax.axhline(0,color="#8b979b",lw=.7)
balance_ax.set_yscale("symlog",linthresh=2)
balance_ax.set_yticks([-2,0,2,5,15,50],labels=["−2","0","2","5","15","50"])
balance_ax.set(ylabel="Cooling excess (% of input)",
               xlabel=r"Time in eddy turnovers  ($t_{\rm eddy}=0.440$)")
balance_ax.set_xscale("symlog",linthresh=4)
balance_ax.set_xticks([0,2,4,8,16,32,64,90],labels=["0","2","4","8","16","32","64","90"])
balance_ax.set_xlim(0,91)
dist_ax.tick_params(labelbottom=False)
for ax in (spec_ax,dist_ax,balance_ax):
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y",alpha=.22)
    ax.set_axisbelow(True)
fig.suptitle(r"Spectral spin-up and energy balance in the $32^3$ driven box",fontsize=14)
fig.savefig(root/"settling.png",dpi=180,facecolor="white")
for name in runs:
    print(name, json.dumps({k:report[name][k] for k in ("spectrum","energy","joint")}, indent=2))
