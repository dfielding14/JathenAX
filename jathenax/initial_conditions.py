"""Periodic test problems on the unit domain centered at the origin."""

import numpy as np


ACOUSTIC_AMPLITUDE = 5e-4


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


def initial_state(problem, coordinates, eos, gamma, cs, axis="x", *, initial_sound_speed=1.0):
    """Conserved fields on periodic cells, with velocities in physical x, y, z order."""
    ndim = len(coordinates)
    shape = tuple(len(x) for x in reversed(coordinates))
    grid = np.meshgrid(*reversed(coordinates), indexing="ij", sparse=True)[::-1]
    rho = np.ones(shape)
    velocity = [0.0] * ndim
    pressure = rho / gamma
    scalars = []
    if problem in ("acoustic", "sod"):
        if axis not in tuple("xyz"[:ndim]):
            raise ValueError("problem.axis must be an active direction")
        normal = "xyz".index(axis)
        x = grid[normal]
        if problem == "acoustic":
            perturbation = ACOUSTIC_AMPLITUDE * np.sin(2.0 * np.pi * x)
            rho = np.broadcast_to(1.0 + perturbation, shape)
            sound = 1.0 if eos == "ideal" else cs
            velocity[normal] = sound * perturbation
            pressure = 1.0 / gamma + perturbation
        else:
            if eos != "ideal":
                raise ValueError("the Sod problem requires an ideal gas")
            rho = np.broadcast_to(np.where(x < 0, 1.0, 0.125), shape)
            pressure = np.where(x < 0, 1.0, 0.1)
    elif problem == "kh":
        if ndim < 2:
            raise ValueError("Kelvin–Helmholtz shear layers require x and y")
        layer = kelvin_helmholtz(*coordinates[:2])
        velocity[:2] = [layer[..., i] / layer[..., 0] for i in (1, 2)]
        scalars = [np.broadcast_to(layer[..., 3], shape)]
    elif problem == "smooth":
        phases = [2.0 * np.pi * x for x in grid]
        rho = 1.0 + 0.1 * np.sin(sum(phases))
        velocity[0] = 0.2 if ndim == 1 else 0.2 * np.cos(phases[1])
        if ndim >= 2:
            velocity[1] = 0.2 * np.sin(phases[0])
        if ndim == 3:
            velocity[2] = 0.2 * np.cos(phases[0] + phases[1])
        pressure = rho / gamma
    elif problem == "turbulence" and eos == "ideal":
        if not np.isfinite(initial_sound_speed) or initial_sound_speed <= 0:
            raise ValueError("the initial adiabatic sound speed must be finite and positive")
        pressure = rho * initial_sound_speed**2 / gamma
    elif problem != "turbulence":
        raise ValueError(f"unknown problem {problem!r}")

    fields = [rho, *(rho * v for v in velocity)]
    if eos == "ideal":
        fields.append(pressure / (gamma - 1.0) + 0.5 * rho * sum(v**2 for v in velocity))
    return np.stack(fields + scalars, axis=-1)
