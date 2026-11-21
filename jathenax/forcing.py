"""AthenaK's continuous, Cartesian parabolic driving for periodic hydro."""

from fractions import Fraction
from itertools import product
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


def cartesian_modes(nlow, nhigh, kx_bounds, ky_bounds=None, kz_bounds=None):
    """Enumerate the signed Cartesian modes in x-outermost order."""
    if not 1 <= nlow < nhigh:
        raise ValueError("parabolic driving requires 1 <= nlow < nhigh")
    bounds = [b for b in (kx_bounds, ky_bounds, kz_bounds) if b is not None]
    modes = [k for k in product(*(range(lo, hi + 1) for lo, hi in bounds))
             if nlow**2 <= sum(i*i for i in k) <= nhigh**2]
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
    ndim = len(lengths)
    if ndim == 1 and sol_fraction == 1.0:
        raise ValueError("1D driving needs a compressive component (sol_fraction < 1)")
    dk = 2.0 * pi / np.asarray(lengths)
    k = modes * dk
    kiso = np.sqrt(np.sum(k**2, axis=1))
    kpeak = npeak * 2.0 * pi / lengths[0]
    band = nhigh * max(dk) - nlow * min(dk)
    # Match the reference build's fused multiply-add here. At the band edge, rounding the
    # product before adding one changes the amplitude at order sqrt(epsilon).
    prefactor = Fraction(float(-4.0 / band**2))
    power = np.asarray([
        float(prefactor * Fraction(float(offset_sq)) + 1)
        for offset_sq in (kiso - kpeak)**2
    ])
    norm = np.sqrt(np.abs(power))
    norm *= (kpeak / kiso)**((ndim - 1) / 2)

    correlation = exp(-dt_update / tcorr) if tcorr > 1e-6 else 0.0
    noise_weight = sqrt(1.0 - correlation * correlation)
    gaussian = ran_gaussian_st(rseed)
    amplitudes = np.empty((updates, len(modes), ndim, 2))
    for update in range(updates):
        # Draw even at the endpoints of the band where the spectral weight is zero.
        noise = np.fromiter(gaussian, dtype=float, count=2 * ndim * len(modes)).reshape(-1, ndim, 2)
        noise *= norm[:, None, None]
        k_dot_noise = np.sum(k[:, :, None] * noise, axis=1)
        longitudinal = k[:, :, None] * k_dot_noise[:, None] / kiso[:, None, None]**2
        noise = sol_fraction * (noise - longitudinal) + (1.0 - sol_fraction) * longitudinal
        amplitudes[update] = (
            noise if update == 0 else correlation * amplitudes[update - 1] + noise_weight * noise
        )
    return amplitudes


def periodic_basis(modes, coordinates, lengths):
    """Tabulate separable Fourier phase factors in x, y, z order."""
    k = modes * (2.0 * pi / jnp.asarray(lengths, dtype=coordinates[0].dtype))
    factors = []
    for i, x in enumerate(coordinates):
        phase = k[:, i, None] * x
        factors.extend((jnp.cos(phase), jnp.sin(phase)))
    return tuple(factors)


def forcing_basis(modes, x, y, lengths):
    return periodic_basis(modes, (x, y), lengths)


def render_force(amplitudes, basis):
    """Sum the seeded Fourier modes with a full-precision matrix multiply."""
    ndim = amplitudes.shape[1]
    cx, sx = basis[:2]
    real, imag = amplitudes[..., 0], amplitudes[..., 1]
    if ndim == 1:
        phase = jnp.concatenate((cx, -sx), axis=0).T
        return jnp.matmul(phase, jnp.concatenate((real, imag)), precision=jax.lax.Precision.HIGHEST)

    # Separate x from the transverse coordinates, avoiding a full box for every mode.
    modes, nx = cx.shape
    cx, sx = cx[..., None], sx[..., None]
    real, imag = real[:, None, :], imag[:, None, :]
    weights = jnp.concatenate((cx*real - sx*imag, sx*real + cx*imag), axis=0)
    weights = weights.reshape(2*modes, nx*ndim)

    def contract(cosine, sine):
        phase = jnp.concatenate((cosine, -sine), axis=0).reshape(2*modes, -1).T
        field = jnp.matmul(phase, weights, precision=jax.lax.Precision.HIGHEST)
        return field.reshape(*cosine.shape[1:], nx, ndim)

    cy, sy = basis[2:4]
    if ndim == 2:
        return contract(cy, sy)
    cz, sz = basis[4:6]
    if modes > max(128, nx):
        # For broad forcing bands, bound workspace by contracting one xy plane at a time.
        def xy_plane(z):
            c, s = z[0][:, None], z[1][:, None]
            return contract(c*cy - s*sy, c*sy + s*cy)
        return jax.lax.map(xy_plane, (cz.T, sz.T))
    return contract(cz[:, :, None]*cy[:, None, :] - sz[:, :, None]*sy[:, None, :],
                    cz[:, :, None]*sy[:, None, :] + sz[:, :, None]*cy[:, None, :])


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


def normalize_force(U, raw_force, dt, dedt, *, return_parameters=False):
    """Remove net momentum input and solve for the specified kinetic power."""
    rho = U[..., :1]
    ndim = U.ndim - 1
    momentum = U[..., 1:1 + ndim]
    mean = jnp.sum(rho * raw_force, axis=tuple(range(ndim))) / jnp.sum(rho)
    force = raw_force - mean
    m0 = jnp.mean(0.5 * rho[..., 0] * jnp.sum(force**2, axis=-1) * dt)
    m1 = jnp.mean(jnp.sum(momentum * force, axis=-1))
    # Before driving starts the zero kick must also have a finite, zero derivative.
    off = dedt == 0.0
    root = jnp.sqrt(jnp.where(off, 1.0, m1*m1 + 4.0*m0*dedt))
    numerator = jnp.where(m1 >= 0.0, 2.0 * dedt, root - m1)
    denominator = jnp.where(m1 >= 0.0, m1 + root, 2.0 * m0)
    scale = numerator / jnp.where(off, 1.0, denominator)
    scale = jnp.where(off, 0.0, jnp.where(m0 > 1e-30, scale, jnp.nan))
    acceleration = scale * force
    return (acceleration, scale, mean) if return_parameters else acceleration
