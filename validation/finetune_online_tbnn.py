"""Fine-tune a stress checkpoint through live, driven, 4:1 filtered trajectories."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers
import numpy as np

from jathenax.les import Transport
from jathenax.tbnn import load_weights, save_weights
from validation.online_reference import live_blocks, save_block, training_batch
from validation.train_online_tbnn import adam_proposal, checked_update, finite_difference, training_loss


def train(snapshot, checkpoint, calibration, output, updates):
    started = monotonic()
    fit = json.loads(calibration.read_text())
    with np.load(snapshot, allow_pickle=False) as f:
        cfg = json.loads(str(f["settings"]))
        dtype = jax.dtypes.canonicalize_dtype(f["conserved"].dtype)
    if (cfg["turb_driving"] != fit["driving"] or cfg["hydro"]["gamma"] != fit["gamma"]
            or cfg["hydro_srcterms"]["tau_cool"] != fit["tau_cool"]):
        raise ValueError("training snapshot must match the calibration's seed, driving, EOS and cooling")

    model = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype), load_weights(checkpoint))
    if not np.isclose(model["gamma"], fit["gamma"]):
        raise ValueError("checkpoint EOS differs from the training EOS")
    initial = model.pop("params")
    rate, weight = 0.001, fit["direct_loss_weight"]
    init, _, parameters = optimizers.adam(rate)
    opt = init(initial)
    evaluate, derivatives = training_loss(model, Transport(**fit["transport"]), fit["tau_cool"],
                                          jnp.asarray(fit["loss_scales"], dtype), fit["stress_scale2"])
    timing, executables = {}, {}
    blocks = live_blocks(snapshot, cfg, None, updates, fit["target_dt"], compile_times=timing)
    report = {k: fit[k] for k in ("gamma", "tau_cool", "target_dt", "driving", "fields", "loss_scales",
                                   "stress_scale2", "direct_loss_weight", "transport")}
    report.update(backend=str(jax.devices()[0]), jax_version=jax.__version__, precision=str(dtype),
                  fine_checkpoint=str(snapshot), initial_checkpoint=str(checkpoint), calibration=str(calibration),
                  initial_checkpoint_sha256=sha256(checkpoint.read_bytes()).hexdigest(),
                  initial_optimizer="fresh Adam moments", updates=[])
    for i in range(updates):
        begin = monotonic()
        live, clock = next(blocks)
        times, dts, ends = clock
        targets = (times+dts)[ends]
        keys = ("dt", "duration", "stage", "stress", "final", "impulse", "sgs_work")
        batch = training_batch({k: live[k] for k in keys}, len(ends))
        jax.block_until_ready((live, batch))
        ready = monotonic()
        p = parameters(opt)
        n = len(batch["dt"])
        compile_start = monotonic()
        if i == 0:
            propose = adam_proposal.lower(0, p, opt, rate).compile()
        if n not in executables:
            executables[n] = (evaluate.lower(p, batch, weight).compile(),
                              derivatives.lower(p, batch, weight).compile())
        forward, backward = executables[n]
        compiled = monotonic()

        value, grad = jax.block_until_ready(backward(p, batch, weight))
        differentiated = monotonic()
        opt, result = checked_update(i, opt, batch, value, grad, evaluate=forward, parameters=parameters,
                                     direct_weight=weight, learning_rate=rate, propose=propose)
        jax.block_until_ready(opt)
        accepted = monotonic()
        path = output / f"reference_{i:03d}.npz"
        save_block(path, live, clock)
        written = monotonic()
        fine_compile = timing.pop("fine_compile_seconds", 0.)
        result.update(start=float(times[0]), end=float(targets[-1]), steps=len(dts), padded_steps=n,
                      comparison_times=targets.tolist(), reference=path.name,
                      timing_seconds=dict(reference=ready-begin-fine_compile, fine_compile=fine_compile,
                                          coarse_compile=compiled-compile_start, gradient=differentiated-compiled,
                                          adam_and_postcheck=accepted-differentiated, write=written-accepted,
                                          end_to_end=written-begin))
        report["updates"].append(result)
        print(f"Update {i+1}/{updates} t={targets[-1]:.2f}, {len(dts)} steps: "
              f"loss {result['before']['loss']:.7f} -> {result['after']['loss']:.7f}, "
              f"fine {ready-begin-fine_compile:.2f}s, compile {fine_compile+compiled-compile_start:.2f}s, "
              f"coarse train {accepted-compiled:.2f}s, end-to-end {written-begin:.2f}s", flush=True)
        if i == 0:
            report.update(fine_n=cfg["mesh"]["nx"], coarse_n=batch["stage"].shape[2])
        if i in (0, updates-1):
            result["gradient_check"] = finite_difference(p, batch, forward, backward, weight, dtype)
            result["gradient_check_seconds"] = monotonic()-written
        del live
        if i+1 < updates:
            del batch

    final = dict(model, params=parameters(opt))
    path = output / "stress_tbnn.npz"
    save_weights(path, final)
    restored = load_weights(path)
    for a, b in zip(jax.tree_util.tree_leaves(final), jax.tree_util.tree_leaves(restored)):
        np.testing.assert_array_equal(a, b)
    reloaded, _ = forward(restored["params"], batch, weight)
    np.testing.assert_array_equal(float(reloaded), result["after"]["loss"])
    report.update(checkpoint_sha256=sha256(path.read_bytes()).hexdigest(), checkpoint_reload_loss=float(reloaded),
                  wall_seconds=monotonic()-started)
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot", "checkpoint", "calibration", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--updates", type=int, default=10)
    args = parser.parse_args()
    if args.updates < 1 or args.output.exists():
        parser.error("supply a positive update count and a new output directory")
    jax.config.update("jax_enable_x64", jax.default_backend().lower() != "metal")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as directory:
        out = Path(directory) / "result"
        out.mkdir()
        train(args.snapshot, args.checkpoint, args.calibration, out, args.updates)
        out.rename(args.output)
    print(args.output.resolve(), flush=True)


if __name__ == "__main__":
    main()
