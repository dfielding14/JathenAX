"""Fit the stress-only TBNN and its two controls on the 32-step pilot window."""

import argparse
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers
import numpy as np
from scipy.optimize import least_squares

from jathenax.input import read_input
from jathenax.tbnn import (FROBENIUS, load_weights, predict_stress, raw_stress,
                          realizability_scale, save_weights, stress_inputs)


TRAIN, TEST = slice(0, 20), slice(24, 32)
RK3 = np.array([1/6, 1/6, 2/3])
FROB = np.array(FROBENIUS)
K0 = 1e-8
SEED, UPDATES, BATCH = 73, 1200, 2048


def read_window(directory):
    """Load coarse states, target deviators, cycle numbers and fine-step times."""
    files = sorted(directory.glob("step_*.npz"))
    if len(files) != 32:
        raise ValueError("the pilot split requires 32 consecutive RK3 steps")
    states, stresses, cycles, starts, dts = [], [], [], [], []
    for path in files:
        with np.load(path, allow_pickle=False) as f:
            if (f["stage_state"].shape != (3, 8, 8, 8, 7)
                    or f["stage_stress"].shape != (3, 8, 8, 8, 6) or f["filter_factor"] != 4):
                raise ValueError(f"{path}: expected a 32-to-8 RK3 archive")
            states.append(f["stage_state"])
            stresses.append(f["stage_stress"])
            cycles.append(int(f["cycle"]))
            starts.append(float(f["time"]))
            dts.append(float(f["dt"]))
    state, full = np.stack(states), np.stack(stresses)
    starts, dts = np.array(starts), np.array(dts)
    if (not np.all(np.diff(cycles) == 1) or np.any(dts <= 0)
            or not np.allclose(starts[1:], (starts + dts)[:-1], rtol=0, atol=1e-12)
            or not np.isfinite(state).all() or not np.isfinite(full).all()
            or np.any(state[..., [0, 5]] <= 0) or np.any(state[..., 6] < 0)):
        raise ValueError("invalid coarse state or noncontiguous reference steps")
    trace = full[..., 0] + full[..., 3] + full[..., 5]
    np.testing.assert_allclose(trace, 2 * state[..., 6], rtol=0, atol=2e-13)
    full[..., [0, 3, 5]] -= trace[..., None] / 3
    return state, full, np.array(cycles), starts, dts


def on_grids(fn, states, *args):
    # Keep the same compiled grid shape for offline fitting and checkpoint verification.
    run = jax.jit(fn)
    return jax.tree_util.tree_map(lambda *x: np.stack(x),
                                  *[jax.device_get(run(*args, s.astype(np.float32))) for s in states])


def inner(a, b):
    return np.sum(a * b * FROB, axis=-1)


def smallest_eigenvalue(dev, energy):
    xx, xy, xz, yy, yz, zz = np.moveaxis(dev.astype(np.float64), -1, 0)
    matrix = np.stack((xx, xy, xz, xy, yy, yz, xz, yz, zz), axis=-1)
    return np.linalg.eigvalsh(matrix.reshape(*xx.shape, 3, 3))[..., 0] + 2*energy/3


def limit_on_host(tensor, energy):
    # NumPy's independent eigensolve also checks the analytic Metal-compatible limiter.
    eig = smallest_eigenvalue(tensor, energy)
    bound = 2*energy/3
    scale = np.where(eig < 0, bound/np.where(eig < 0, bound-eig, 1.), 1.)
    return tensor * scale[..., None], eig, scale


def fit(inputs, target, weight, gamma):
    # Only the caller's training slice is visible to the optimizer and both controls.
    x = {k: v.reshape(-1, *v.shape[5:]) for k, v in inputs.items()}
    y, w = target.reshape(-1, 6), weight.ravel()
    unit = x["unit_eddy"].astype(np.float64)
    c_nu = max(0., np.sum(w * inner(unit, y)) / np.sum(w * inner(unit, unit)))
    columns = (2*x["K"])[:, None, None] * x["basis"]
    metric = w[:, None] * FROB
    design = columns.transpose(0, 2, 1) * np.sqrt(metric)[..., None]
    constants = np.linalg.lstsq(design.reshape(-1, 2),
                                ((y - c_nu*unit) * np.sqrt(metric)).ravel(), rcond=None)[0]
    def constant_residual(coeff):
        raw = c_nu*unit + np.einsum("i,nij->nj", coeff, columns)
        limited, _, _ = limit_on_host(raw, x["K"].astype(np.float64))
        return ((limited - y) * np.sqrt(metric)).ravel()

    control = least_squares(constant_residual, constants)
    if not control.success:
        raise RuntimeError(control.message)
    constants = control.x
    f = x["features"].astype(np.float64)
    mean = np.average(f, axis=0, weights=w)
    scale = np.sqrt(np.average((f - mean)**2, axis=0, weights=w))
    if np.any(scale == 0):
        raise ValueError("constant feature on the pilot training block")
    rng = np.random.default_rng(SEED)
    params = dict(w1=rng.normal(0, 1/np.sqrt(f.shape[-1]), (f.shape[-1], 16)),
                  b1=np.zeros(16), w2=np.zeros((16, 2)), b2=np.zeros(2))
    weights = jax.tree_util.tree_map(lambda a: jnp.asarray(a, dtype=jnp.float32),
                                     dict(feature_mean=mean, feature_scale=scale,
                                          gamma=gamma, k0=K0, c_nu=c_nu, params=params))
    # The expected minibatch loss is the full RK-time-weighted squared relative error.
    normalized_w = (w / np.mean(w * inner(y, y))).astype(np.float32)
    y = y.astype(np.float32)
    def loss(params, batch, truth, sample_weight):
        pred = raw_stress(dict(weights, params=params), batch)
        pred *= realizability_scale(pred, batch["K"])[..., None]
        return jnp.mean(sample_weight * jnp.sum((pred - truth)**2 * jnp.asarray(FROB, truth.dtype), -1))

    init, update, get = optimizers.adam(lambda i: 0.008 / (1 + i/600))
    @jax.jit
    def step(i, opt, batch, truth, sample_weight):
        value, grad = jax.value_and_grad(loss)(get(opt), batch, truth, sample_weight)
        norm = sum(jnp.sum(g*g) for g in jax.tree_util.tree_leaves(grad))
        return update(i, grad, opt), value, norm

    opt = init(weights["params"])
    initial = np.asarray(raw_stress(weights, {k: v[:512] for k, v in x.items()}))
    np.testing.assert_allclose(initial, float(weights["c_nu"]) * x["unit_eddy"][:512], rtol=1e-6, atol=1e-7)
    losses, norms = [], []
    then = time.perf_counter()
    for i in range(UPDATES):
        idx = rng.integers(len(y), size=BATCH)
        opt, value, norm = step(i, opt, {k: v[idx] for k, v in x.items()}, y[idx], normalized_w[idx])
        losses.append(value)
        norms.append(norm)
    losses, norms = np.asarray(losses), np.sqrt(np.asarray(norms))
    if not np.isfinite(losses).all() or not np.isfinite(norms).all() or norms[0] == 0:
        raise FloatingPointError("non-finite optimization or zero initial weight gradient")
    weights["params"] = get(opt)
    details = dict(seed=SEED, adam_updates=UPDATES, minibatch=BATCH, hidden=16,
                   parameter_count=sum(p.size for p in get(opt).values()),
                   gradient_norm_first=float(norms[0]), gradient_norm_last=float(norms[-1]),
                   all_update_gradients_finite=True, training_seconds=time.perf_counter()-then)
    return weights, constants, details


def score(pred, truth, w, strain, energy, raw_min_eigenvalue, scale):
    weighted = lambda a: float(np.sum(w * a) / np.sum(w))
    power = weighted(inner(truth, truth))
    production, reference = -inner(pred, strain), -inner(truth, strain)
    return dict(relative_rmse=np.sqrt(weighted(inner(pred-truth, pred-truth))/power),
                tensor_cosine=weighted(inner(pred, truth)) / np.sqrt(weighted(inner(pred, pred))*power),
                mean_deviatoric_production=weighted(production), backscatter_fraction=weighted(production < 0),
                production_relative_rmse=np.sqrt(weighted((production-reference)**2)/weighted(reference**2)),
                raw_non_psd_fraction=weighted(raw_min_eigenvalue < 0),
                raw_min_eigenvalue=float(np.min(raw_min_eigenvalue)),
                largest_fractional_limiter_correction=float(1 - np.min(scale)),
                limited_min_eigenvalue=float(np.min(smallest_eigenvalue(pred, energy))))


def plot_result(path, times, dts, errors, targets, predictions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(9, 6.2), layout="constrained")
    gs = fig.add_gridspec(2, 3, height_ratios=(1, 1.05))
    ax = fig.add_subplot(gs[0, :])
    t = times - times[0]
    ax.axvspan(t[20], t[24], color="0.9")
    ax.axvspan(t[24], t[-1]+dts[-1], color="#e3edf4")
    for name, color in zip(errors, ("#bc6828", "#626b71", "#096886")):
        ax.plot(t+dts/2, errors[name], label=name, color=color, lw=2)
    for x, word in ((0., "Fit"), (t[20], "Gap"), (t[24], "Held back")):
        ax.text(x+.0007, 1.015, word, transform=ax.get_xaxis_transform(), fontsize=9)
    ax.set(xlabel="Time since window start (code units)", ylabel="Relative stress RMSE",
           xlim=(0, t[-1]+dts[-1]))
    ax.spines[["right", "top"]].set_visible(False)
    ax.legend(ncol=3, fontsize=9, loc="center left")
    sections = [a[-1, 0, 4, :, :, 1] for a in (targets, predictions["Eddy viscosity"], predictions["TBNN"])]
    limit = np.max(np.abs(sections))
    axes = [fig.add_subplot(gs[1, i]) for i in range(3)]
    for panel, field, title in zip(axes, sections, ("Filtered reference", "Eddy viscosity", "TBNN")):
        im = panel.imshow(field, origin="lower", cmap="RdBu_r", vmin=-limit, vmax=limit,
                          extent=(-.5, .5, -.5, .5), interpolation="nearest")
        panel.set(title=title, xlabel="x", xticks=(-.5, 0, .5), yticks=(-.5, 0, .5))
    axes[0].set_ylabel("y")
    fig.colorbar(im, ax=axes, shrink=.88, label=r"$\tau_{xy}$ (code pressure)")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def train(directory, output):
    state, truth, cycles, times, dts = read_window(directory)
    gamma = read_input(directory.parent / "input.toml")["hydro"]["gamma"]
    w = np.broadcast_to(dts[:, None, None, None, None] * RK3[None, :, None, None, None], state.shape[:-1])
    inputs = on_grids(lambda s: stress_inputs(s, gamma, K0), state)
    weights, constants, training = fit({k: v[TRAIN] for k, v in inputs.items()}, truth[TRAIN], w[TRAIN], gamma)
    output.mkdir(parents=True, exist_ok=True)
    pred = on_grids(predict_stress, state, weights)
    eddy = float(weights["c_nu"])*inputs["unit_eddy"]
    raw = {"Eddy viscosity": eddy,
           "Constant coefficients": eddy + (2*inputs["K"])[..., None] * np.sum(constants[:, None]*inputs["basis"], axis=-2),
           "TBNN": on_grids(lambda m, s: raw_stress(m, stress_inputs(s, m["gamma"], m["k0"])), state, weights)}
    cs = np.sqrt(gamma*(gamma-1)*state[..., 5]/state[..., 0])
    # Unit box, Delta = 1/N. Isotropic pressure-dilatation is outside this target.
    strain = inputs["basis"][..., 0, :] * (cs * state.shape[-2])[..., None]
    energy = state[..., 6]
    report = dict(window=directory.name, backend=str(jax.devices()[0]), jax_version=jax.__version__,
                  tensor_convention="positive Favre stress, trace removed, full Frobenius norm",
                  weighting="fine dt times SSPRK3 quadrature, equal cell volumes",
                  gamma=gamma, k0=K0, c_nu=float(weights["c_nu"]), constant_coefficients=constants.tolist(),
                  training=training, scores={}, splits={})
    idx = {"train": TRAIN, "gap": slice(20,24), "test": TEST}
    for key, s in idx.items():
        measured = -inner(truth[s], strain[s])
        report["splits"][key] = dict(cycles=[int(cycles[s][0]), int(cycles[s][-1])],
                                     time_interval=[float(times[s][0]), float((times+dts)[s][-1])],
                                     cell_stage_samples=int(w[s].size),
                                     measured_deviatoric_production=float(np.average(measured, weights=w[s])),
                                     measured_backscatter_fraction=float(np.average(measured < 0, weights=w[s])))
    predictions, errors = {}, {}
    for name, tensor in raw.items():
        limited, eig, scale = limit_on_host(tensor, energy)
        if name == "TBNN":
            np.testing.assert_allclose(pred, limited, rtol=2e-4, atol=3e-6)
            limited = pred
        predictions[name] = limited
        report["scores"][name] = {k: score(limited[s], truth[s], w[s], strain[s], energy[s], eig[s], scale[s])
                                  for k, s in idx.items() if k != "gap"}
        errors[name] = np.sqrt(np.sum(w * inner(limited-truth, limited-truth), axis=(1,2,3,4)) /
                               np.sum(w * inner(truth, truth), axis=(1,2,3,4)))
    if report["scores"]["TBNN"]["train"]["relative_rmse"] >= report["scores"]["Eddy viscosity"]["train"]["relative_rmse"]:
        raise RuntimeError("supervised stress fit did not improve on its initialization")
    save_weights(output / "stress_tbnn.npz", weights)
    np.savez_compressed(output / "predictions_before_reload.npz", stress=pred)
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    plot_result(output / "stress_comparison.png", times, dts, errors, truth, predictions)
    print(json.dumps(dict(backend=report["backend"], training=training, scores=report["scores"]), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("window", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    train(args.window, args.output)
