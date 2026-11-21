"""Differentiable short trajectories for the first-derivative stress model."""

import jax
import jax.numpy as jnp

from .les import RK3_WEIGHTS, apply_impulse, conserved, filtered_state, les_step, les_timestep
from .tbnn import FROBENIUS, predict_stress


def physical(fields):
    return jnp.concatenate((fields[..., :1], fields[..., 1:4] / fields[..., :1],
                            fields[..., 5:7]), axis=-1)


def rollout(model, data, transport, tau_cool, *, stress="tbnn", substeps=1, initial=None,
            return_sources=False):
    """Initialize once. Observed endpoints never enter the coarse state update."""
    if substeps < 1:
        raise ValueError("substeps must be positive")
    q0 = conserved(data["stage"][0, 0] if initial is None else initial)
    dx = 1 / q0.shape[0]
    if stress == "measured" and substeps != 1:
        raise ValueError("measured tensors are recorded at the original fine RK stage times")

    def advance(q, inputs):
        dt, impulse, sgs_work, *measured = inputs
        before = q[..., :5].mean(axis=(0, 1, 2))
        cooling, fraction, minima = 0., 0., jnp.full((3,), jnp.inf, q.dtype)
        sub_dt = dt * (1/substeps)
        for _ in range(substeps):
            bound = les_timestep(q, dx, model, transport, tau_cool)
            fraction = jnp.maximum(fraction, sub_dt/bound)
            q, sink, low = les_step(q, sub_dt, dx, model, transport, tau_cool,
                                    stress=stress, measured=measured[0] if measured else None)
            cooling, minima = cooling + sink, jnp.minimum(minima, low)
        q, work = apply_impulse(q, impulse, sgs_work)
        end = filtered_state(q)
        minima = jnp.minimum(minima, jnp.stack([end[..., i].min() for i in (0, 5, 6)]))
        source = jnp.concatenate((jnp.zeros(1, q.dtype), impulse.mean(axis=(0, 1, 2)),
                                  (work.mean() - cooling)[None]))
        residual = q[..., :5].mean(axis=(0, 1, 2)) - before - source
        outputs = (end, residual, minima, fraction)
        if return_sources:
            outputs += (jnp.stack((work.mean(), cooling)),)
        return q, outputs

    inputs = (data["dt"], data["impulse"], data["sgs_work"])
    if stress == "measured":
        inputs += (data["stress"],)
    # Recompute RK intermediates from step-boundary states during reverse differentiation.
    # This also gives the correct transpose on Metal when the step contains limiter masks.
    _, history = jax.lax.scan(jax.checkpoint(advance, prevent_cse=False), q0, inputs)
    return history


def trajectory_errors(history, data, scales):
    error = (physical(history) - physical(data["final"])) / scales
    by_field = jnp.mean(error**2, axis=(1, 2, 3))
    by_field = jnp.sum(by_field * data["duration"][:, None], axis=0) / data["duration"].sum()
    return jnp.stack((by_field[0], by_field[1:4].mean(), by_field[4], by_field[5]))


def objective(params, fixed, data, transport, tau_cool, scales, stress_scale2, direct_weight):
    model = dict(fixed, params=params)
    states, residuals, minima, fractions = rollout(model, data, transport, tau_cool)
    fields = trajectory_errors(states, data, scales)
    prediction = predict_stress(model, data["stage"])
    square = ((prediction - data["stress"])**2 * jnp.asarray(FROBENIUS, states.dtype)).sum(-1)
    weights = data["dt"][:, None] * jnp.asarray(RK3_WEIGHTS, states.dtype)
    direct = (weights * square.mean(axis=(2, 3, 4))).sum() / (data["dt"].sum() * stress_scale2)
    return fields.mean() + direct_weight * direct, dict(
        fields=fields, direct=direct, budget=jnp.max(jnp.abs(residuals), axis=0),
        minima=minima.min(axis=0), step_fraction=fractions.max())
