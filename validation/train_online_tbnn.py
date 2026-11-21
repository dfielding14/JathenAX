"""Verify a saved stress rollout and train on a live 32^3 to 8^3 reference."""

import argparse
from functools import partial
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers
import numpy as np

from jathenax.input import read_input
from jathenax.les import (GAMMA, RK3_WEIGHTS, Transport, face_average, les_rhs, unit_transport)
from jathenax.online import objective, physical, rollout, trajectory_errors
from jathenax.tbnn import FROBENIUS, load_weights, save_weights
from validation.online_reference import live_blocks, read_saved, training_batch


def calibrate(data, model):
    """Fit the analytic remainder and loss scales on the original 20 training steps."""
    n = 20
    fields, tau = data["stage"][:n], data["stress"][:n]
    dt = data["dt"][:n]
    weights = dt[:, None] * jnp.asarray(RK3_WEIGHTS, fields.dtype)
    dx = 1 / fields.shape[-2]

    def moments(s, stress, heat, third):
        fit = jnp.zeros((2, 2), dtype=s.dtype)
        for normal in range(3):
            axis = 2-normal
            h, k = unit_transport(s, model["k0"], axis)
            q = face_average(heat[..., normal], axis)
            fk = face_average(third[..., normal], axis) + ((GAMMA-1)/GAMMA)*q
            fit = fit + jnp.stack((jnp.stack(((h*q).mean(), (h*h).mean())),
                                    jnp.stack(((k*fk).mean(), (k*k).mean()))))
        production = -les_rhs(s, stress, dx, model, Transport(0., 0., 0.))[..., 5].mean()
        dissipation = (s[..., 6] * jnp.sqrt(s[..., 6]/s[..., 0])).mean() / dx
        return fit, production, dissipation

    evaluate = jax.jit(jax.vmap(jax.vmap(moments)))
    fit, production, dissipation = evaluate(fields, tau, data["heat"][:n], data["third"][:n])
    sums = (weights[..., None, None] * fit).sum(axis=(0, 1))
    coeff = jnp.maximum(sums[:, 0] / sums[:, 1], 0.)
    # Periodic flux divergences integrate to zero. This is an effective K-budget fit.
    delta_k = (data["final"][n-1, ..., 6]-fields[0, 0, ..., 6]).mean()
    injected = data["sgs_work"][:n].mean(axis=(1, 2, 3)).sum()
    decay = jnp.maximum(((weights*production).sum() + injected - delta_k)
                        / (weights*dissipation).sum(), 0.)
    z = physical(fields)
    spatial_axes = (2, 3, 4)
    variance = ((z-z.mean(axis=spatial_axes, keepdims=True))**2).mean(axis=spatial_axes)
    variance = (weights[..., None]*variance).sum(axis=(0, 1)) / dt.sum()
    scales = jnp.sqrt(variance.at[1:4].set(variance[1:4].mean()))
    norm2 = (tau**2 * jnp.asarray(FROBENIUS, fields.dtype)).sum(-1).mean(axis=spatial_axes)
    stress_scale2 = (weights*norm2).sum() / dt.sum()
    if not all(np.isfinite(x).all() for x in (coeff, decay, scales, stress_scale2)) or (
            np.min(scales) <= 0 or stress_scale2 <= 0):
        raise ValueError("degenerate training data for the analytic closures or loss scales")
    return Transport(coeff[0], coeff[1], decay), scales, stress_scale2


def checked(value, dtype):
    loss, aux = value
    minima, budget = np.asarray(aux["minima"]), np.asarray(aux["budget"])
    if (not np.isfinite(float(loss)) or not np.isfinite(minima).all() or min(minima[:2]) <= 0
            or minima[2] < 0 or not 0 < float(aux["step_fraction"]) <= 1
            or not np.all(np.isfinite(budget)) or max(budget) > 100*np.finfo(dtype).eps):
        raise FloatingPointError(f"invalid coarse trajectory: loss={loss}, minima={minima}, budget={budget}")
    return dict(loss=float(loss), trajectory_mse=float(aux["fields"].mean()),
                field_rmse=np.sqrt(np.asarray(aux["fields"])).tolist(), direct_mse=float(aux["direct"]),
                max_budget_residual=budget.tolist(), min_density_thermal_K=minima.tolist(),
                max_step_over_estimate=float(aux["step_fraction"]))


def flat(tree):
    return np.concatenate([np.ravel(v) for v in jax.tree_util.tree_leaves(tree)])


def parameter_step(params, direction, length):
    return jax.tree_util.tree_map(lambda p, d: p + length*d, params, direction)


def training_loss(model, transport, tau_cool, scales, stress_scale2):
    def loss(params, data, weight):
        return objective(params, model, data, transport, tau_cool, scales, stress_scale2, weight)
    return jax.jit(loss), jax.jit(jax.value_and_grad(loss, has_aux=True))


@jax.jit
def adam_proposal(number, grad, previous, rate):
    _, update, _ = optimizers.adam(rate)
    return update(number, grad, previous)


def checked_update(number, previous, data, value, grad, *, evaluate, parameters, direct_weight,
                   learning_rate=0.001, propose=adam_proposal):
    dtype = data["dt"].dtype
    before = checked(value, dtype)
    gradient = flat(grad)
    norm = float(np.linalg.norm(gradient))
    if not np.isfinite(norm) or norm <= 0:
        raise FloatingPointError("invalid weight gradient")
    for attempt in range(8):
        rate = learning_rate * 0.5**attempt
        new = propose(number, grad, previous, rate)
        try:
            after = checked(evaluate(parameters(new), data, direct_weight), dtype)
        except FloatingPointError:
            continue
        # Adam's accumulated momentum can raise the loss on an individual new trajectory.
        change = flat(parameters(new)) - flat(parameters(previous))
        return new, dict(before=before, after=after, learning_rate=rate, gradient_norm=norm,
                         weight_change=float(np.linalg.norm(change)),
                         gradient_dot_update=float(np.dot(gradient, change)))
    raise FloatingPointError("Adam update failed trajectory admissibility at every attempted rate")


def finite_difference(params, data, evaluate, derivatives, direct_weight, dtype):
    # The trajectory-only derivative isolates the path through all three RK stages.
    (_, _), grad = derivatives(params, data, 0.)
    norm = np.linalg.norm(flat(grad))
    if not np.isfinite(norm) or norm == 0:
        raise FloatingPointError("zero or nonfinite trajectory gradient")
    direction = jax.tree_util.tree_map(lambda g: g / norm, grad)
    (_, _), full_grad = derivatives(params, data, direct_weight)
    analytic = np.array([norm, np.dot(flat(full_grad), flat(direction))])
    report = []
    steps = (1e-4, 3e-5, 1e-5) if dtype == np.float64 else (1e-2, 4e-3, 1.5e-3)
    for h in steps:
        def losses(sign):
            loss, aux = evaluate(parameter_step(params, direction, sign*h), data, direct_weight)
            return np.array([float(aux["fields"].mean()), float(loss)])
        measured = (losses(1) - losses(-1)) / (2*h)
        relative = np.abs(measured-analytic)/np.maximum(np.abs(analytic), 1e-12)
        report.append(dict(h=h, centered=measured.tolist(), relative_error=relative.tolist()))
    tolerance = 0.001 if dtype == np.float64 else 0.01
    if sum(max(row["relative_error"]) < tolerance for row in report) < 2:
        raise AssertionError(f"trajectory gradient disagrees with finite differences: {analytic}, {report}")
    return dict(autodiff=analytic.tolist(), probes=report)


def train(pilot, out, snapshot, live_updates, cadence):
    window, = pilot.glob("window_[0-9]*")
    data, clock = read_saved(window)
    dtype = data["dt"].dtype
    cfg = read_input(pilot / "input.toml")
    tc = cfg["hydro_srcterms"]["tau_cool"]
    model = load_weights(pilot / "stress_fit" / "stress_tbnn.npz")
    if not np.isclose(model["gamma"], GAMMA):
        raise ValueError("the checkpoint must use gamma=5/3")
    model = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype), model)
    initial_params = model.pop("params")
    transport, scales, stress_scale2 = calibrate(data, model)
    direct_weight, learning_rate = 0.01, 0.001

    evaluate, derivatives = training_loss(model, transport, tc, scales, stress_scale2)
    init, _, parameters = optimizers.adam(learning_rate)
    opt = init(initial_params)
    update_once = partial(checked_update, evaluate=evaluate, parameters=parameters,
                          direct_weight=direct_weight, learning_rate=learning_rate)

    @partial(jax.jit, static_argnames=("stress", "substeps"))
    def control(params, d, stress, substeps):
        history, budget, lows, fractions = rollout(dict(model, params=params), d, transport, tc,
                                                  stress=stress, substeps=substeps)
        errors = trajectory_errors(history, d, scales)
        return history, dict(fields=errors, direct=0., budget=jnp.abs(budget).max(axis=0),
                             minima=lows.min(axis=0), step_fraction=fractions.max())

    report = dict(backend=str(jax.devices()[0]), jax_version=jax.__version__, precision=str(dtype),
                  gamma=GAMMA, tau_cool=tc, target_dt=cadence,
                  archive=str(window), fine_checkpoint=str(snapshot), driving=cfg["turb_driving"],
                  fields=["density", "velocity", "thermal_energy", "subgrid_kinetic_energy"],
                  loss_scales=np.asarray(scales).tolist(), stress_scale2=float(stress_scale2),
                  direct_loss_weight=direct_weight,
                  transport={k: float(v) for k, v in transport._asdict().items()},
                  calibration_steps=20, controls={}, updates=[])
    np.testing.assert_allclose(data["stage"][1:, 0], data["final"][:-1], rtol=1e-6, atol=1e-7)
    times, dts, ends = clock
    comparisons = dict(time=(times+dts)[ends], reference=np.asarray(data["final"])[ends])
    for name, substeps in (("measured", 1), ("analytic", 1), ("tbnn", 1), ("analytic", 2), ("tbnn", 2)):
        states, aux = control(initial_params, data, name, substeps)
        summary = checked((aux["fields"].mean(), aux), dtype)
        key = name if substeps == 1 else name + "_half_dt"
        report["controls"][key] = {k: v for k, v in summary.items() if k not in ("loss", "direct_mse")}
        comparisons[key] = np.asarray(states)[ends]
        print(f"{key} trajectory MSE {summary['trajectory_mse']:.7g}", flush=True)
    report["gradient_check"] = finite_difference(initial_params, data, evaluate, derivatives, direct_weight, dtype)
    disk_value, disk_grad = derivatives(initial_params, data, direct_weight)
    disk_opt, disk_update = update_once(0, opt, data, disk_value, disk_grad)
    report["saved_update"] = disk_update
    print(f"Saved trajectory objective {disk_update['before']['loss']:.7g} -> {disk_update['after']['loss']:.7g}",
          flush=True)

    for number, (live, live_clock) in enumerate(live_blocks(snapshot, cfg, clock, live_updates, cadence)):
        starts, deltas, ids = live_clock
        targets = (starts+deltas)[ids]
        if number:
            np.testing.assert_allclose(targets/cadence, np.rint(targets/cadence), rtol=0, atol=1e-10)
        # Retain a compact, directly replayable training block including every matched kick.
        path = out / f"reference_{number:03d}.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, **jax.device_get(live), time=starts, physical_dt=deltas,
                                endpoints=ids, target_time=targets, target_state=np.asarray(live["final"])[ids])
        batch = training_batch(live, len(ids))
        started = monotonic()
        value, grad = derivatives(parameters(opt), batch, direct_weight)
        new_opt, result = update_once(number, opt, batch, value, grad)
        initial = np.asarray(live["stage"][0, 0])
        t_delta = (1 / initial.shape[0]) / np.sqrt(initial[..., 6].mean() / initial[..., 0].mean())
        result.update(start=float(starts[0]), end=float(targets[-1]), steps=len(starts),
                      comparison_times=targets.tolist(), duration_over_t_delta=float(deltas.sum()/t_delta),
                      update_seconds=monotonic()-started)
        if number == 0:
            # Test the IO path at identical precision, and separately compare the independent fine rerun.
            with np.load(path, allow_pickle=False) as f:
                reread = {k: jnp.asarray(f[k]) for k in live}
            reread = training_batch(reread, len(ids))
            reload_value, reload_grad = derivatives(initial_params, reread, direct_weight)
            reload_opt, _ = update_once(0, opt, reread, reload_value, reload_grad)
            for a, b in ((value, reload_value), (grad, reload_grad), (new_opt, reload_opt)):
                np.testing.assert_array_equal(flat(a), flat(b))
            errors = {k: float(jnp.max(jnp.abs(live[k]-data[k]))) for k in live}
            tolerance = 3e-11 if dtype == np.float64 else 5e-5
            if max(errors.values()) > tolerance:
                raise AssertionError(f"live fine simulation diverged from archived inputs: {errors}")
            gradient_error = float(np.linalg.norm(flat(grad)-flat(disk_grad))/np.linalg.norm(flat(disk_grad)))
            update_error = float(np.max(np.abs(flat(parameters(new_opt))-flat(parameters(disk_opt)))))
            if abs(float(value[0]-disk_value[0])) > tolerance or gradient_error > 0.005 or update_error > tolerance:
                raise AssertionError(f"archive/live update mismatch: gradient {gradient_error}, weights {update_error}")
            report["saved_live"] = dict(same_records_bitwise=True, independent_archive_max_abs=errors,
                                         loss_abs=float(abs(value[0]-disk_value[0])),
                                         gradient_relative_l2=gradient_error, weights_max_abs=update_error)
            states, _ = control(parameters(new_opt), live, "tbnn", 1)
            comparisons["after_first_update"] = np.asarray(states)[ids]
            states, aux = control(parameters(new_opt), live, "tbnn", 2)
            result["half_dt_trajectory_mse"] = checked((aux["fields"].mean(), aux), dtype)["trajectory_mse"]
            comparisons["after_first_update_half_dt"] = np.asarray(states)[ids]
            np.savez_compressed(out / "first_trajectory.npz", **comparisons)
        else:
            result["controls_before_training"] = {}
            for name in ("analytic", "tbnn"):
                _, aux = control(initial_params, batch, name, 1)
                baseline = checked((aux["fields"].mean(), aux), dtype)
                result["controls_before_training"][name] = baseline["trajectory_mse"]
        opt = new_opt
        report["updates"].append(result)
        print(f"Live update {number+1} at t={result['end']:.8f}: "
              f"{result['before']['loss']:.7g} -> {result['after']['loss']:.7g}", flush=True)
    final = dict(model, params=parameters(opt))
    save_weights(out / "stress_tbnn.npz", final)
    restored = load_weights(out / "stress_tbnn.npz")
    # Pass all restored checkpoint fields into the same forward operator.
    check_forward = jax.jit(lambda m, d: rollout(m, d, transport, tc)[0])
    ref = check_forward(final, batch)
    reloaded = check_forward(restored, batch)
    np.testing.assert_array_equal(ref, reloaded)
    report["checkpoint_reload_max_abs"] = float(jnp.max(jnp.abs(ref-reloaded)))
    (out / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--snapshot", type=Path, help="stationary fine checkpoint listed in source.json")
    parser.add_argument("--live-updates", type=int, default=2,
                        help="additional live updates after reproducing the saved update")
    parser.add_argument("--target-dt", type=float, default=0.01, help="spacing of new live comparison times")
    args = parser.parse_args()
    if args.output.exists() or args.live_updates < 1 or not 0 < args.target_dt < np.inf:
        parser.error("supply a new output directory, at least one new live update and a positive target interval")
    jax.config.update("jax_enable_x64", jax.default_backend().lower() != "metal")
    snapshot = args.snapshot or Path(json.loads((args.pilot / "source.json").read_text())["snapshot"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as tmp:
        result = Path(tmp) / "result"
        result.mkdir()
        train(args.pilot, result, snapshot, args.live_updates, args.target_dt)
        result.rename(args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
