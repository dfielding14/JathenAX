"""Periodic test problems on the unit square centered at the origin."""

import numpy as np


def kelvin_helmholtz(x, y):
    """Return (rho, rho*vx, rho*vy, rho*tracer) for two periodic shear layers."""
    x, y = np.meshgrid(x, y)
    distance = np.abs(y) - 0.25
    transition = np.clip(-distance / 0.025, -1.0, 1.0)
    tracer = 0.5 + transition * (0.75 - 0.25 * transition**2)
    rho = np.ones_like(tracer)
    vx = tracer - 0.5
    vy = 0.02 * np.sin(4 * np.pi * x) * np.exp(-(distance / 0.05)**2)
    return np.stack((rho, rho * vx, rho * vy, rho * tracer), axis=-1)
