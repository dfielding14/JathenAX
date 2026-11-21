"""Compare frozen stress closures on the same independent driving realization."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import jax
import jax.numpy as jnp
import numpy as np

from jathenax.les import GAMMA, Transport, analytic_stress
from jathenax.online import rollout, trajectory_errors
from jathenax.tbnn import FROBENIUS, load_weights, predict_stress
from validation.online_reference import live_blocks, save_block, training_batch


def save_json(path, data):
    body = json.dumps(data, indent=2, allow_nan=False) + "\n"
    with path.open("x") as f:
        f.write(body)


def record_reference(snapshot, output, blocks=31):
    """Save eight round-time targets per block and all intervening driving kicks."""
    with np.load(snapshot, allow_pickle=False) as f:
        cfg = json.loads(str(f["settings"]))
        start = float(f["time"])
    cadence = 0.01
    if blocks < 1:
        raise ValueError("record at least one block")
    if output.exists():
        raise FileExistsError(output)
    with TemporaryDirectory(dir=output.parent) as temp:
        directory = Path(temp) / "reference"
        directory.mkdir()
        manifest = dict(snapshot=str(snapshot.resolve()), start=start, cadence=cadence,
                        driving=cfg["turb_driving"], tau_cool=cfg["hydro_srcterms"]["tau_cool"],
                        gamma=cfg["hydro"]["gamma"], short_blocks=list(range(0, blocks, 6)), blocks=[])
        previous = start
        for i, (data, clock) in enumerate(live_blocks(snapshot, cfg, None, blocks, cadence)):
            times, dt, ends = clock
            target_time = (times + dt)[ends]
            if i == 0:
                manifest.update(fine_n=cfg["mesh"]["nx"], coarse_n=data["final"].shape[1])
            if not previous < target_time[0] <= previous + cadence + 2e-13:
                raise ValueError("first target must follow the previous state on the comparison clock")
            np.testing.assert_allclose(times[0], previous, rtol=0, atol=2e-13)
            expected = (round(target_time[0]/cadence) + np.arange(8)) * cadence
            np.testing.assert_allclose(target_time, expected, rtol=0, atol=2e-13)
            previous = target_time[-1]
            path = directory / f"block_{i:03d}.npz"
            save_block(path, data, clock)
            manifest["blocks"].append(dict(file=path.name, sha256=sha256(path.read_bytes()).hexdigest()))
        save_json(directory / "reference.json", manifest)
        directory.rename(output)


def means(fields):
    """Space means of density, total energy, COM-frame resolved kinetic energy, U and K."""
    q = np.asarray(fields, dtype=np.float64)
    avg = lambda x: x.mean(axis=(-3, -2, -1))
    rho, mom = q[..., 0], q[..., 1:4]
    bulk = avg(0.5 * np.sum(mom*mom, axis=-1) / rho)
    bulk -= 0.5 * np.sum(mom.mean(axis=(-4, -3, -2))**2, axis=-1) / avg(rho)
    return np.stack((avg(rho), avg(q[..., 4]), bulk, avg(q[..., 5]), avg(q[..., 6])), axis=-1)


def long_summary(times, history, work, cooling):
    t, h = np.asarray(times), np.asarray(history)
    duration = t[-1] - t[0]
    density, total, bulk, thermal, sgs = np.sum((h[:-1]+h[1:])*0.5*np.diff(t)[:, None], axis=0)/duration
    return dict(duration=float(duration), mean_density=float(density), mean_energy=float(total),
                mean_thermal=float(thermal), mean_K=float(sgs), mean_resolved_kinetic=float(bulk),
                mach=float(np.sqrt(2*(bulk+sgs)/(GAMMA*(GAMMA-1)*thermal))),
                resolved_mach=float(np.sqrt(2*bulk/(GAMMA*(GAMMA-1)*thermal))),
                driving_power=float(work/duration), cooling_power=float(cooling/duration),
                storage_rate=float((h[-1, 1]-h[0, 1])/duration),
                energy_residual_per_time=float((h[-1, 1]-h[0, 1]-work+cooling)/duration))


def evaluate(reference, calibration, checkpoint, output, *, analytic=False, continuous=False, substeps=1):
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((reference / "reference.json").read_text())
    fit = json.loads(calibration.read_text())
    forcing = manifest["driving"]
    if (forcing["rseed"] == fit["driving"]["rseed"] or manifest["gamma"] != fit["gamma"]
            or manifest["tau_cool"] != fit["tau_cool"]
            or any(v != fit["driving"].get(k) for k, v in forcing.items() if k != "rseed")):
        raise ValueError("use a separate reference seed with the training physics")
    model = load_weights(checkpoint)
    transport = Transport(**fit["transport"])
    tau, scales = fit["tau_cool"], jnp.asarray(fit["loss_scales"])
    stress = "analytic" if analytic else "tbnn"

    @jax.jit
    def advance(initial, kicks, endpoints):
        states, budgets, minima, step_fraction, sources = rollout(
            model, kicks, transport, tau, stress=stress, initial=initial,
            substeps=substeps, return_sources=True)
        return states[endpoints], budgets, minima, step_fraction, sources

    @jax.jit
    def short_score(states, truth, intervals, moment_state, measured):
        fields = trajectory_errors(states, dict(final=truth, duration=intervals), scales)
        pred = analytic_stress(model, moment_state) if analytic else predict_stress(model, moment_state)
        metric = jnp.asarray(FROBENIUS, truth.dtype)
        def tensor_power(x):
            return (intervals * (metric*x*x).sum(-1).mean(axis=(1, 2, 3))).sum() / intervals.sum()
        return fields, tensor_power(pred-measured), tensor_power(measured)

    short, times, history, fine_history = [], [], [], []
    coarse_work, fine_work = np.zeros(2), np.zeros(2)
    carried, failures = None, []
    low, budget_max, max_fraction = np.full(3, np.inf), np.zeros(5), 0.
    for i, block in enumerate(manifest["blocks"]):
        if not continuous and i not in manifest["short_blocks"]:
            continue
        path = reference / block["file"]
        if sha256(path.read_bytes()).hexdigest() != block["sha256"]:
            raise ValueError(f"reference checksum changed in {path}")
        with np.load(path, allow_pickle=False) as f:
            initial = f["initial"] if carried is None else carried
            dtype = initial.dtype
            kicks = {k: jnp.asarray(f[k], dtype) for k in ("dt", "impulse", "sgs_work")}
            kicks = training_batch(kicks, len(f["endpoints"]))
            states, budgets, minima, fractions, sources = jax.device_get(
                advance(initial, kicks, f["endpoints"]))
            size = len(f["dt"])
            minima, fractions, budgets = minima[:size], fractions[:size], budgets[:size]
            valid = (np.isfinite(minima).all(axis=1) & (minima[:, :2] > 0).all(axis=1)
                     & (minima[:, 2] >= 0) & (fractions > 0) & (fractions <= 1)
                     & (np.abs(budgets).max(axis=1) <= 100*np.finfo(dtype).eps))
            if not valid.all():
                k = int(np.flatnonzero(~valid)[0])
                failures.append(dict(block=i, step=k, time=float(f["time"][k] + f["dt"][k]),
                                     reason="stage positivity, finiteness, timestep or conservation check"))
                if continuous:
                    break
                continue
            low = np.minimum(low, minima.min(axis=0))
            budget_max = np.maximum(budget_max, np.abs(budgets).max(axis=0))
            max_fraction = max(max_fraction, float(fractions.max()))
            t, start = f["target_time"], float(f["time"][0])
            if continuous:
                if carried is None:
                    times.append(start)
                    history.append(means(initial))
                    fine_history.append(means(f["initial"]))
                times.extend(t)
                history.extend(means(states))
                fine_history.extend(means(f["target"]))
                coarse_work += sources[:size].sum(axis=0, dtype=np.float64)
                fine_work += [f[k].sum(dtype=np.float64) for k in ("drive_work", "cooling_loss")]
                carried = states[-1]
            else:
                intervals = np.diff(np.r_[start, t]).astype(dtype)
                err, stress_error, stress_norm = short_score(states, f["target"], intervals,
                                                             f["moment_state"], f["moment_stress"])
                short.append(dict(start=start, end=float(t[-1]), steps=size,
                                  field_mse=np.asarray(err).tolist(), mse=float(err.mean()),
                                  stress_error_power=float(stress_error), stress_power=float(stress_norm)))

    report = dict(seed=forcing["rseed"], training_seed=fit["driving"]["rseed"], backend=str(jax.devices()[0]),
                  checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint.read_bytes()).hexdigest(),
                  calibration=str(calibration), reference=str(reference), stress=stress, continuous=continuous,
                  substeps=substeps, failures=failures, fields=fit["fields"],
                  min_density_thermal_K=low.tolist() if np.isfinite(low).all() else None,
                  max_abs_budget_residual=budget_max.tolist(), max_step_fraction=max_fraction)
    if continuous and len(times) > 1:
        report.update(coarse=long_summary(times, history, *coarse_work),
                      fine=long_summary(times, fine_history, *fine_work), time=times,
                      history=np.asarray(history).tolist(), fine_history=np.asarray(fine_history).tolist(),
                      history_fields=["density", "total_energy", "resolved_kinetic", "thermal", "K"])
    elif not continuous and short:
        by_field = np.mean([r["field_mse"] for r in short], axis=0)
        error = sum(r["stress_error_power"] for r in short)
        power = sum(r["stress_power"] for r in short)
        report.update(starts=short, field_rmse=np.sqrt(by_field).tolist(),
                      mean_mse=float(by_field.mean()), direct_stress_relative_rmse=float(np.sqrt(error/power)))
    save_json(output, report)
    print(output, "failed checks" if failures else "complete", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    record = commands.add_parser("reference", help="save held-out targets and matched driving")
    record.add_argument("snapshot", type=Path)
    record.add_argument("output", type=Path)
    record.add_argument("--blocks", type=int, default=31, help="number of eight-target blocks")
    score = commands.add_parser("score", help="score one frozen closure")
    for key in ("reference", "calibration", "checkpoint", "output"):
        score.add_argument(key, type=Path)
    score.add_argument("--analytic", action="store_true", help="use the checkpoint's analytic eddy viscosity")
    score.add_argument("--continuous", action="store_true", help="initialize once for the full reference")
    score.add_argument("--substeps", type=int, default=1, help="hydro steps per recorded kick")
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", jax.default_backend().lower() != "metal")
    if args.command == "reference":
        record_reference(args.snapshot, args.output, args.blocks)
    else:
        evaluate(args.reference, args.calibration, args.checkpoint, args.output, analytic=args.analytic,
                 continuous=args.continuous, substeps=args.substeps)


if __name__ == "__main__":
    main()
