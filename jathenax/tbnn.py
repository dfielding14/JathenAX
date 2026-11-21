"""First-derivative, stress-only TBNN for the periodic cubic Form A grid."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import jax.numpy as jnp
import numpy as np


COMPONENTS = ("xx", "xy", "xz", "yy", "yz", "zz")
FEATURES = ("k/cs^2", "tr(G)", "Sdev:Sdev", "Omega:Omega")
BASES = ("dev(sym(G))", "dev(G G^T)")
FROBENIUS = (1., 2., 2., 1., 2., 1.)
FORMAT = "jathenax.stress_tbnn.v1"


def deviatoric(packed):
    """Trace removal in the positive Favre momentum-flux convention."""
    third = (packed[..., 0] + packed[..., 3] + packed[..., 5]) / 3
    return packed - third[..., None] * jnp.asarray((1, 0, 0, 1, 0, 1), dtype=packed.dtype)


def pack(tensor):
    return jnp.stack([tensor[..., i, j] for i in range(3) for j in range(i, 3)], axis=-1)


def stress_inputs(state, gamma, k0):
    """Build local inputs from (R, mx, my, mz, Ebar, U, K) on (..., z, y, x, 7)."""
    if state.ndim < 4 or state.shape[-1] != 7 or len(set(state.shape[-4:-1])) != 1:
        raise ValueError("expected a cubic grid of seven filtered fields")
    rho, thermal, energy = state[..., 0], state[..., 5], state[..., 6]
    velocity = state[..., 1:4] / rho[..., None]
    cs2 = gamma * (gamma - 1) * thermal / rho
    cs, k = jnp.sqrt(cs2), energy / rho
    # In G = Delta * grad(v) / cs the cubic cell width cancels the stencil denominator.
    grad = jnp.stack([(jnp.roll(velocity, -1, axis) - jnp.roll(velocity, 1, axis)) * 0.5
                      for axis in (-2, -3, -4)], axis=-1) / cs[..., None, None]
    transpose = jnp.swapaxes(grad, -1, -2)
    strain = deviatoric(pack(0.5 * (grad + transpose)))
    rotation = 0.5 * (grad - transpose)
    quadratic = deviatoric(pack(grad @ transpose))
    features = jnp.stack((k / cs2, jnp.trace(grad, axis1=-2, axis2=-1),
                          (strain**2 * jnp.asarray(FROBENIUS, dtype=state.dtype)).sum(axis=-1),
                          (rotation**2).sum(axis=(-1, -2))), axis=-1)
    vk = k / (jnp.sqrt(k + k0) + jnp.sqrt(k0))
    unit_eddy = (-2 * rho * cs * vk)[..., None] * strain
    return dict(features=features, basis=jnp.stack((strain, quadratic), axis=-2),
                unit_eddy=unit_eddy, K=energy)


def realizability_scale(stress, energy):
    """Radially limit the deviator so 2K/3 I + stress is positive semidefinite."""
    # For a traceless symmetric 3x3 matrix the eigenvalues solve a scalar cubic.
    # This expression also runs on Metal, which has no JAX eigh lowering.
    p2 = (stress**2 * jnp.asarray(FROBENIUS, dtype=stress.dtype)).sum(axis=-1) / 6
    p = jnp.sqrt(jnp.where(p2 > 0, p2, 1.))
    unit = stress / p[..., None]
    # Slice components in place. Transposing this batch corrupts its reverse gradient on Metal.
    xx, xy, xz, yy, yz, zz = (unit[..., i] for i in range(6))
    r = 0.5 * (xx*(yy*zz - yz*yz) - xy*(xy*zz - xz*yz) + xz*(xy*yz - yy*xz))
    eps = jnp.finfo(stress.dtype).eps
    angle = jnp.arccos(jnp.clip(r, -1 + eps, 1 - eps)) / 3
    negative_min = jnp.where(r <= -1 + eps, 2*p, -2*p*jnp.cos(angle + 2*jnp.pi/3))
    negative_min = jnp.where(p2 > 0, negative_min, 0.)
    # Leave a roundoff margin when the full tensor lands on the PSD boundary.
    bound = (2/3) * energy * (1 - 16*eps)
    return jnp.where(negative_min > bound, bound / jnp.where(negative_min > 0, negative_min, 1.), 1.)


def raw_stress(weights, inputs):
    """Analytic eddy viscosity plus two dimensionless learned coefficients."""
    x = (inputs["features"] - weights["feature_mean"]) / weights["feature_scale"]
    params = weights["params"]
    hidden = jnp.tanh(x @ params["w1"] + params["b1"])
    coeff = hidden @ params["w2"] + params["b2"]
    correction = (coeff[..., None] * inputs["basis"]).sum(axis=-2)
    return weights["c_nu"] * inputs["unit_eddy"] + (2*inputs["K"])[..., None] * correction


def predict_stress(weights, state):
    """Predict packed deviatoric stress from the coarse state in code pressure units."""
    inputs = stress_inputs(state, weights["gamma"], weights["k0"])
    raw = raw_stress(weights, inputs)
    return raw * realizability_scale(raw, inputs["K"])[..., None]


def save_weights(path, weights):
    metadata = dict(format=FORMAT, features=FEATURES, bases=BASES, components=COMPONENTS,
                    stencil="periodic centered first derivatives on a cubic grid",
                    state="R,mx,my,mz,Ebar,U,K", units="code pressure, positive Favre stress",
                    activation="tanh", limiter="radial full-stress realizability")
    path = Path(path)
    scalars = {k: weights[k] for k in ("gamma", "k0", "c_nu", "feature_mean", "feature_scale")}
    with TemporaryDirectory(dir=path.parent) as directory:
        tmp = Path(directory) / "weights.npz"
        np.savez(tmp, metadata=json.dumps(metadata), **scalars, **weights["params"])
        tmp.replace(path)


def load_weights(path):
    """Load the parameters, fitted scaling, and analytic coefficient for inference."""
    with np.load(path, allow_pickle=False) as f:
        meta = json.loads(str(f["metadata"].item()))
        if (meta["format"] != FORMAT or tuple(meta["features"]) != FEATURES
                or tuple(meta["bases"]) != BASES or tuple(meta["components"]) != COMPONENTS):
            raise ValueError("incompatible stress TBNN checkpoint")
        params = {k: f[k] for k in ("w1", "b1", "w2", "b2")}
        weights = {k: f[k] for k in ("gamma", "k0", "c_nu", "feature_mean", "feature_scale")}
    n, h = len(FEATURES), params["b1"].size
    shapes = ((n, h), (h,), (h, len(BASES)), (len(BASES),))
    if (h == 0 or any(v.shape != shape for v, shape in zip(params.values(), shapes))
            or any(weights[k].shape != (n,) for k in ("feature_mean", "feature_scale"))
            or any(weights[k].shape != () for k in ("gamma", "k0", "c_nu"))
            or any(not np.isfinite(v).all() for v in (*params.values(), *weights.values()))
            or np.any(weights["feature_scale"] <= 0) or weights["gamma"] <= 1
            or weights["k0"] <= 0 or weights["c_nu"] < 0):
        raise ValueError("invalid stress TBNN parameter shape, value, or normalization")
    return dict(**weights, params=params)
