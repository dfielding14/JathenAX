"""Reload a fitted pilot checkpoint and check inference on the archived grids."""

import argparse
from itertools import permutations, product
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jathenax.tbnn import FROBENIUS, load_weights, predict_stress
from validation.train_stress_tbnn import on_grids, read_window, smallest_eigenvalue


def check_symmetries(model, state):
    predict = jax.jit(predict_stress)
    ref = np.asarray(predict(model, state))
    xyz = np.indices(state.shape[:3])[::-1]
    rows, cols = np.triu_indices(3)
    packed = np.array([[0, 1, 2], [1, 3, 4], [2, 4, 5]])
    error = 0.
    for p in permutations(range(3)):
        p = np.array(p)
        for signs in product((1, -1), repeat=3):
            sign = np.array(signs)
            # Q maps old axis p[i] to new axis i about the cell-centered box origin.
            old = np.empty_like(xyz)
            old[p] = np.where(sign[:, None, None, None] > 0, xyz, state.shape[0]-1-xyz)
            index = tuple(old[::-1])
            transformed = state[index].copy()
            transformed[..., 1:4] = transformed[..., 1:4][..., p] * sign
            expected = ref[index][..., packed[p[rows], p[cols]]] * (sign[rows]*sign[cols])
            error = max(error, float(np.max(np.abs(predict(model, transformed) - expected))))
    boost = np.array((1.75, -2.25, 0.5), dtype=state.dtype)
    shifted = state.copy()
    shifted[..., 4] += state[..., 1:4] @ boost + 0.5*state[..., 0]*sum(boost**2)
    shifted[..., 1:4] += state[..., :1] * boost
    boost_err = float(np.max(np.abs(predict(model, shifted) - ref)))
    if max(error, boost_err) > 3e-5*np.max(np.abs(ref)) + 2e-6:
        raise AssertionError(f"failed symmetry or Galilean boost check: {error}, {boost_err}")
    cold_sgs = state.copy()
    cold_sgs[..., 4] -= cold_sgs[..., 6]
    cold_sgs[..., 6] = 0
    np.testing.assert_array_equal(predict(model, cold_sgs), 0)
    return dict(cubic_symmetries=48, cubic_max_abs_error=error, boost_max_abs_error=boost_err,
                zero_K_max_abs_stress=0.)


def check_gradient(model, state, truth):
    def objective(params):
        err = predict_stress(dict(model, params=params), state) - truth
        f = jnp.asarray(FROBENIUS, dtype=truth.dtype)
        return jnp.sum(err**2 * f) / jnp.sum(truth**2 * f)

    loss = jax.jit(objective)
    params = model["params"]
    grad = jax.jit(jax.grad(objective))(params)
    norm = float(jnp.sqrt(sum(jnp.sum(g*g) for g in grad.values())))
    if not np.isfinite(norm) or norm == 0:
        raise AssertionError("invalid checkpoint weight gradient")
    h = 0.002
    plus, minus = ({k: v + shift*grad[k]/norm for k, v in params.items()}
                   for shift in (h, -h))
    fd = float((loss(plus) - loss(minus))/(2*h))
    np.testing.assert_allclose(fd, norm, rtol=0.02, atol=2e-4)
    return dict(weight_gradient_norm=norm, directional_finite_difference=fd)


def verify(directory, output):
    model = load_weights(output / "stress_tbnn.npz")
    state, truth, _, _, _ = read_window(directory)
    after = on_grids(predict_stress, state, model)
    with np.load(output / "predictions_before_reload.npz", allow_pickle=False) as f:
        before = f["stress"]
    backend = str(jax.devices()[0])
    trained_on = json.loads((output / "metrics.json").read_text())["backend"]
    if backend == trained_on:
        np.testing.assert_array_equal(after, before)
    else:
        np.testing.assert_allclose(after, before, atol=3e-6, rtol=3e-5)
    max_trace = float(np.max(np.abs(np.sum(after[..., [0, 3, 5]], axis=-1))))
    min_eig = float(np.min(smallest_eigenvalue(after, state[..., 6])))
    if min_eig < -3e-6 or max_trace > 3e-6:
        raise AssertionError(f"inadmissible predicted stress: trace error {max_trace}, min eigenvalue {min_eig}")
    result = dict(backend=backend, prediction_shape=list(after.shape), cell_stage_predictions=after[..., 0].size,
                  bitwise_equal=bool(np.array_equal(after, before)), reload_max_abs_error=float(np.max(np.abs(after-before))),
                  max_abs_deviatoric_trace=max_trace, min_full_stress_eigenvalue=min_eig)
    result.update(check_symmetries(model, state[-1, 0].astype(np.float32)))
    result.update(check_gradient(model, state[0, 0].astype(np.float32), truth[0, 0].astype(np.float32)))
    print(json.dumps(result, indent=2, allow_nan=False))
    (output / f"verify_{jax.devices()[0].platform.lower()}.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("window", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    verify(args.window, args.output)
