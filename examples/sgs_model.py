"""Example SGS predictor with a constant kinematic eddy viscosity."""

import jax.numpy as jnp
import numpy as np


def load_weights(checkpoint):
    """Load on the host. Replace this with your model's checkpoint reader."""
    if checkpoint is None:
        return {"nu": np.float32(1e-3)}
    with np.load(checkpoint, allow_pickle=False) as saved:
        nu = np.asarray(saved["nu"], dtype=np.float32)
    if nu.ndim != 0 or not np.isfinite(nu) or nu < 0:
        raise ValueError("Expected a finite nonnegative scalar 'nu' in the checkpoint")
    return {"nu": nu}


def predict_stress(weights, primitives):
    """Map the whole periodic (rho, vx, vy) grid to (tau_xx, tau_xy, tau_yy)."""
    ny, nx, _ = primitives.shape
    rho = primitives[..., 0]
    velocity = primitives[..., 1:3]
    # Centered derivatives at cell centers on the example's periodic unit square.
    dv_dx = (jnp.roll(velocity, -1, axis=1) - jnp.roll(velocity, 1, axis=1)) * (0.5 * nx)
    dv_dy = (jnp.roll(velocity, -1, axis=0) - jnp.roll(velocity, 1, axis=0)) * (0.5 * ny)
    dvx_dx, dvy_dy = dv_dx[..., 0], dv_dy[..., 1]
    shear = dv_dx[..., 1] + dv_dy[..., 0]
    trace = (2.0 / 3.0) * (dvx_dx + dvy_dy)
    strain = jnp.stack((2.0 * dvx_dx - trace, shear, 2.0 * dvy_dy - trace), axis=-1)
    # A diffusive stress is negative in the momentum-flux convention.
    return -(rho * weights["nu"])[..., None] * strain
