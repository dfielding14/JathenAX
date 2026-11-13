"""AthenaK's continuous, Cartesian parabolic driving for periodic 2D hydro."""

from fractions import Fraction
from math import exp, log, pi, sqrt

import jax
import jax.numpy as jnp
import numpy as np


def ran_st(seed):
    """AthenaK's shuffled two-generator uniform stream."""
    modulus1, modulus2 = 2147483563, 2147483399
    state1 = state2 = max(1, seed)
    shuffle = [0] * 32
    # Python integers can multiply directly without Schrage's overflow trick.
    for index in reversed(range(40)):
        state1 = (40014 * state1) % modulus1
        if index < 32:
            shuffle[index] = state1
    iy = shuffle[0]
    divisor = 1 + (modulus1 - 1) // 32
    largest = 1.0 - np.finfo(float).eps
    while True:
        state1 = (40014 * state1) % modulus1
        state2 = (40692 * state2) % modulus2
        index = iy // divisor
        iy = shuffle[index] - state2
        shuffle[index] = state1
        if iy < 1:
            iy += modulus1 - 1
        yield min((1.0 / modulus1) * iy, largest)


def ran_gaussian_st(seed):
    """Polar Box–Muller, with AthenaK's return order for each pair."""
    uniform = ran_st(seed)
    while True:
        v1 = 2.0 * next(uniform) - 1.0
        v2 = 2.0 * next(uniform) - 1.0
        radius_sq = v1 * v1 + v2 * v2
        if 0.0 < radius_sq < 1.0:
            scale = sqrt(-2.0 * log(radius_sq) / radius_sq)
            yield v2 * scale
            yield v1 * scale


def cartesian_modes(nlow, nhigh, kx_bounds, ky_bounds):
    """Keep the signed modes in AthenaK's x-outer, y-inner order."""
    if not 1 <= nlow < nhigh:
        raise ValueError("parabolic driving requires 1 <= nlow < nhigh")
    modes = [
        (kx, ky)
        for kx in range(kx_bounds[0], kx_bounds[1] + 1)
        for ky in range(ky_bounds[0], ky_bounds[1] + 1)
        if nlow**2 <= kx**2 + ky**2 <= nhigh**2
    ]
    if not modes:
        raise ValueError("no Cartesian forcing modes in the requested band")
    return np.asarray(modes)


def modal_coefficients(modes, lengths, nlow, nhigh, npeak, sol_fraction, rseed,
                       tcorr, dt_update, updates):
    """Precompute the OU table in float64, indexed by update, mode, dir, real/imag."""
    if not nlow <= npeak <= nhigh or not 0.0 <= sol_fraction <= 1.0:
        raise ValueError("npeak must be in the forcing band and sol_fraction in [0, 1]")
    if not np.isfinite(tcorr) or tcorr < 0.0 or not 0.0 < dt_update < np.inf:
        raise ValueError("tcorr must be finite and nonnegative and dt_update positive")
    Lx, _ = lengths
    dk = 2.0 * pi / np.asarray(lengths)
    k = modes * dk
    kiso = np.sqrt(k[:, 0]**2 + k[:, 1]**2)
    kpeak = npeak * 2.0 * pi / Lx
    band = nhigh * max(dk) - nlow * min(dk)
    # Match the reference build's fused multiply-add here. At the band edge, rounding the
    # product before adding one changes the amplitude at order sqrt(epsilon).
    prefactor = Fraction(float(-4.0 / band**2))
    power = np.asarray([
        float(prefactor * Fraction(float(offset_sq)) + 1)
        for offset_sq in (kiso - kpeak)**2
    ])
    norm = np.sqrt(np.abs(power))
    norm *= np.sqrt(kpeak / kiso)

    correlation = exp(-dt_update / tcorr) if tcorr > 1e-6 else 0.0
    noise_weight = sqrt(1.0 - correlation * correlation)
    gaussian = ran_gaussian_st(rseed)
    amplitudes = np.empty((updates, len(modes), 2, 2))
    for update in range(updates):
        # Draw even at the endpoints of the band where the spectral weight is zero.
        noise = np.fromiter(gaussian, dtype=float, count=4 * len(modes)).reshape(-1, 2, 2)
        noise *= norm[:, None, None]
        k_dot_noise = k[:, 0, None] * noise[:, 0] + k[:, 1, None] * noise[:, 1]
        longitudinal = k[:, :, None] * k_dot_noise[:, None] / kiso[:, None, None]**2
        noise = sol_fraction * (noise - longitudinal) + (1.0 - sol_fraction) * longitudinal
        amplitudes[update] = (
            noise if update == 0 else correlation * amplitudes[update - 1] + noise_weight * noise
        )
    return amplitudes


def forcing_basis(modes, x, y, lengths):
    """Tabulate the two one-dimensional phase factors once on the flow device."""
    k = modes * (2.0 * pi / jnp.asarray(lengths, dtype=x.dtype))
    phase_x = k[:, 0, None] * x
    phase_y = k[:, 1, None] * y
    return jnp.cos(phase_x), jnp.sin(phase_x), jnp.cos(phase_y), jnp.sin(phase_y)


def render_force(amplitudes, basis):
    """Render a raw acceleration in AthenaK's mode order."""
    cos_x, sin_x, cos_y, sin_y = basis
    shape = (cos_y.shape[1], cos_x.shape[1], 2)

    def add_mode(n, force):
        real = cos_x[n] * cos_y[n, :, None] - sin_x[n] * sin_y[n, :, None]
        imag = cos_x[n] * sin_y[n, :, None] + sin_x[n] * cos_y[n, :, None]
        return force + (real[..., None] * amplitudes[n, :, 0]
                        - imag[..., None] * amplitudes[n, :, 1])

    return jax.lax.fori_loop(0, len(amplitudes), add_mode, jnp.zeros(shape, amplitudes.dtype))


def update_force(time, update, raw_force, coefficients, basis, dt_update, tdriv_start):
    """Refresh the cached raw field when the cycle-start time crosses an OU update."""
    requested = jnp.floor((time - tdriv_start) / dt_update).astype(jnp.int32)
    requested = jnp.where(time >= tdriv_start, requested, -1)
    raw_force = jax.lax.cond(
        requested > update,
        lambda: render_force(coefficients.at[requested].get(mode="fill"), basis),
        lambda: raw_force,
    )
    return requested, raw_force


def normalize_force(U, raw_force, dt, dedt):
    """Remove net momentum input and solve for the specified kinetic power."""
    rho = U[..., 0, None]
    momentum = U[..., 1:3]
    mean = jnp.sum(rho * raw_force, axis=(0, 1)) / jnp.sum(rho)
    force = raw_force - mean
    force_sq = force[..., 0]**2 + force[..., 1]**2
    m0 = jnp.mean(0.5 * rho[..., 0] * force_sq * dt)
    m1 = jnp.mean(momentum[..., 0] * force[..., 0] + momentum[..., 1] * force[..., 1])
    root = jnp.sqrt(m1 * m1 + 4.0 * m0 * dedt)
    scale = jnp.where(m1 >= 0.0, 2.0 * dedt / (m1 + root), (root - m1) / (2.0 * m0))
    scale = jnp.where(m0 > 1e-30, scale, jnp.nan)
    return jnp.where(dedt == 0.0, 0.0, scale * force)
