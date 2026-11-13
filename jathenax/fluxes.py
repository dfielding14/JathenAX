"""Face fluxes and their divergence on the periodic grid."""

import jax.numpy as jnp


def dc_face_states(W, axis):
    """Use the two adjacent cell averages at each face."""
    return jnp.roll(W, 1, axis=axis), W


def plm_face_states(W, axis):
    """Reconstruct a piecewise linear profile with AthenaK's harmonic limiter."""
    dl = W - jnp.roll(W, 1, axis=axis)
    dr = jnp.roll(W, -1, axis=axis) - W
    product = dl * dr
    monotone = product > 0.0
    slope = jnp.where(monotone, product / jnp.where(monotone, dl + dr, 1.0), 0.0)
    return jnp.roll(W + slope, 1, axis=axis), W - slope


def ppm4_face_states(W, axis):
    """Return left and right primitives at the low face of each cell."""
    qm2 = jnp.roll(W, 2, axis=axis)
    qm1 = jnp.roll(W, 1, axis=axis)
    qp1 = jnp.roll(W, -1, axis=axis)
    qp2 = jnp.roll(W, -2, axis=axis)

    # Interpolate each cell's low and high edges, then bound them by neighbors.
    low = (7.0 * (W + qm1) - (qm2 + qp1)) / 12.0
    high = (7.0 * (W + qp1) - (qm1 + qp2)) / 12.0
    low = jnp.fmin(jnp.fmax(low, jnp.fmin(W, qm1)), jnp.fmax(W, qm1))
    high = jnp.fmin(jnp.fmax(high, jnp.fmin(W, qp1)), jnp.fmax(W, qp1))

    # Flatten extrema and limit asymmetric parabolas using AthenaK's PPM4 rules.
    dhigh = high - W
    dlow = low - W
    extremum = dhigh * dlow >= 0.0
    low = jnp.where(jnp.abs(dlow) >= 2.0 * jnp.abs(dhigh), W - 2.0 * dhigh, low)
    high = jnp.where(jnp.abs(dhigh) >= 2.0 * jnp.abs(dlow), W - 2.0 * dlow, high)
    low = jnp.where(extremum, W, low)
    high = jnp.where(extremum, W, high)
    return jnp.roll(high, 1, axis=axis), low


def _ppmx_edge(a, b, edge, curvature_a, curvature_b):
    """Limit a parabolic edge that lies outside its neighboring cell averages."""
    curvature = 3.0 * ((a + b) - 2.0 * edge)
    same_sign = (((curvature > 0) & (curvature_a > 0) & (curvature_b > 0))
                 | ((curvature < 0) & (curvature_a < 0) & (curvature_b < 0)))
    magnitude = jnp.fmin(jnp.abs(curvature_a), jnp.abs(curvature_b))
    limited = jnp.copysign(jnp.fmin(1.25 * magnitude, jnp.abs(curvature)), curvature)
    correction = 0.5 * (a + b) - jnp.where(same_sign, limited, 0.0) / 6.0
    return jnp.where((a - edge) * (b - edge) > 0.0, correction, edge)


def ppmx_face_states(W, axis):
    """Reconstruct parabolas with AthenaK's smooth-extremum limiter."""
    qm2 = jnp.roll(W, 2, axis=axis)
    qm1 = jnp.roll(W, 1, axis=axis)
    qp1 = jnp.roll(W, -1, axis=axis)
    qp2 = jnp.roll(W, -2, axis=axis)
    low = (7.0 * (W + qm1) - (qm2 + qp1)) / 12.0
    high = (7.0 * (W + qp1) - (qm1 + qp2)) / 12.0
    d2l = (qm2 + W) - 2.0 * qm1
    d2c = (qm1 + qp1) - 2.0 * W
    d2r = (W + qp2) - 2.0 * qp1
    low = _ppmx_edge(qm1, W, low, d2l, d2c)
    high = _ppmx_edge(W, qp1, high, d2c, d2r)

    dhigh, dlow = high - W, low - W
    extremum = (dhigh * (W - low) <= 0.0) | ((qm1 - W) * (W - qp1) <= 0.0)
    d2 = 6.0 * (low + high - 2.0 * W)
    same_sign = (((d2 > 0) & (d2l > 0) & (d2c > 0) & (d2r > 0))
                 | ((d2 < 0) & (d2l < 0) & (d2c < 0) & (d2r < 0)))
    magnitude = jnp.fmin(jnp.abs(d2c), jnp.fmin(jnp.abs(d2l), jnp.abs(d2r)))
    limited = jnp.copysign(jnp.fmin(1.25 * magnitude, jnp.abs(d2)), d2)
    scale = jnp.fmax(jnp.abs(qm1), jnp.fmax(jnp.abs(W), jnp.abs(qp1)))
    use_curvature = same_sign & (jnp.abs(d2) > 1e-12 * scale)
    ratio = jnp.where(use_curvature, limited / jnp.where(use_curvature, d2, 1.0), 0.0)
    low_monotone = jnp.where(jnp.abs(dlow) >= 2.0 * jnp.abs(dhigh), W - 2.0 * dhigh, low)
    high_monotone = jnp.where(jnp.abs(dhigh) >= 2.0 * jnp.abs(dlow), W - 2.0 * dlow, high)
    low = jnp.where(extremum, W + dlow * ratio, low_monotone)
    high = jnp.where(extremum, W + dhigh * ratio, high_monotone)
    return jnp.roll(high, 1, axis=axis), low


def _wenoz_edge(qm2, qm1, q, qp1, qp2, w0, w1, w2):
    """Combine three candidate polynomials at the high edge of a cell."""
    a0, a1, a2 = 0.1 * w0, 0.6 * w1, 0.3 * w2
    f0 = 2.0 * qm2 - 7.0 * qm1 + 11.0 * q
    f1 = -qm1 + 5.0 * q + 2.0 * qp1
    f2 = 2.0 * q + 5.0 * qp1 - qp2
    return (f0 * a0 + f1 * a1 + f2 * a2) / (6.0 * (a0 + a1 + a2))


def wenoz_face_states(W, axis):
    """Reconstruct fifth-order WENO-Z face values."""
    qm2 = jnp.roll(W, 2, axis=axis)
    qm1 = jnp.roll(W, 1, axis=axis)
    qp1 = jnp.roll(W, -1, axis=axis)
    qp2 = jnp.roll(W, -2, axis=axis)
    b0 = (13.0 / 12.0) * (qm2 + W - 2.0 * qm1)**2 + 0.25 * (qm2 + 3.0 * W - 4.0 * qm1)**2
    b1 = (13.0 / 12.0) * (qm1 + qp1 - 2.0 * W)**2 + 0.25 * (qm1 - qp1)**2
    b2 = (13.0 / 12.0) * (qp2 + W - 2.0 * qp1)**2 + 0.25 * (qp2 + 3.0 * W - 4.0 * qp1)**2
    tau = jnp.abs(b0 - b2)
    # Keep epsilon normal on float32 GPUs and rescale the weights before
    # squaring. The common factor cancels in the weighted average.
    eps = max(1e-42, float(jnp.finfo(W.dtype).tiny))
    d0, d1, d2 = b0 + eps, b1 + eps, b2 + eps
    dmin = jnp.fmin(d0, jnp.fmin(d1, d2))
    scale = jnp.fmax(tau, dmin)
    base = (dmin / scale)**2
    contrast = tau / scale
    w0, w1, w2 = (base + (contrast * (dmin / d))**2 for d in (d0, d1, d2))
    high = _wenoz_edge(qm2, qm1, W, qp1, qp2, w0, w1, w2)
    low = _wenoz_edge(qp2, qp1, W, qm1, qm2, w2, w1, w0)
    return jnp.roll(high, 1, axis=axis), low


RECONSTRUCTIONS = {
    "dc": dc_face_states, "plm": plm_face_states, "ppm4": ppm4_face_states,
    "ppmx": ppmx_face_states, "wenoz": wenoz_face_states,
}


def reconstruct_faces(W, axis, reconstruct, dfloor):
    left, right = RECONSTRUCTIONS[reconstruct](W, axis)
    if reconstruct in ("ppmx", "wenoz"):
        left = left.at[..., 0].set(jnp.fmax(left[..., 0], dfloor))
        right = right.at[..., 0].set(jnp.fmax(right[..., 0], dfloor))
    return left, right


def _roe_flux(rho_l, rho_r, vn_l, vn_r, vt_l, vt_r, cs):
    mn_l, mn_r = rho_l * vn_l, rho_r * vn_r
    mt_l, mt_r = rho_l * vt_l, rho_r * vt_r

    # Work in normal/tangential order and rotate the flux back at the end.
    pressure_l, pressure_r = (cs * cs) * rho_l, (cs * cs) * rho_r
    flux_l = jnp.stack((mn_l, mn_l * vn_l + pressure_l, mn_l * vt_l), axis=-1)
    flux_r = jnp.stack((mn_r, mn_r * vn_r + pressure_r, mn_r * vt_r), axis=-1)
    flux_avg = 0.5 * (flux_l + flux_r)

    # Project the conserved jump onto the two sound waves and the shear wave.
    sqrt_l, sqrt_r = jnp.sqrt(rho_l), jnp.sqrt(rho_r)
    inv_sum = 1.0 / (sqrt_l + sqrt_r)
    vn = (sqrt_l * vn_l + sqrt_r * vn_r) * inv_sum
    vt = (sqrt_l * vt_l + sqrt_r * vt_r) * inv_sum
    wave_minus, wave_plus = vn - cs, vn + cs
    drho, dmn, dmt = rho_r - rho_l, mn_r - mn_l, mt_r - mt_l
    a_minus = drho * (0.5 + 0.5 * vn / cs) - dmn * 0.5 / cs
    a_shear = drho * (-vt) + dmt
    a_plus = drho * (0.5 - 0.5 * vn / cs) + dmn * 0.5 / cs
    c_minus = -0.5 * jnp.abs(wave_minus) * a_minus
    c_shear = -0.5 * jnp.abs(vn) * a_shear
    c_plus = -0.5 * jnp.abs(wave_plus) * a_plus

    mass_flux = flux_avg[..., 0] + c_minus + c_plus
    normal_flux = flux_avg[..., 1] + c_minus * wave_minus + c_plus * wave_plus
    shear_flux = flux_avg[..., 2] + c_minus * vt + c_shear + c_plus * vt
    flux = jnp.stack((mass_flux, normal_flux, shear_flux), axis=-1)

    # Supersonic faces use the upwind physical flux. LLF takes precedence if
    # the Roe intermediate density is negative.
    flux = jnp.where((wave_minus >= 0.0)[..., None], flux_l, flux)
    flux = jnp.where((wave_plus <= 0.0)[..., None], flux_r, flux)
    rho_star = rho_l + a_minus
    use_llf = (rho_star < 0.0) | (rho_star + a_plus < 0.0)
    llf_speed = 0.5 * jnp.fmax(jnp.abs(vn_l) + cs, jnp.abs(vn_r) + cs)
    jump = jnp.stack((drho, dmn, dmt), axis=-1)
    llf = flux_avg - llf_speed[..., None] * jump
    return jnp.where(use_llf[..., None], llf, flux)


def _llf_flux(rho_l, rho_r, vn_l, vn_r, vt_l, vt_r, cs):
    mn_l, mn_r = rho_l * vn_l, rho_r * vn_r
    mt_l, mt_r = rho_l * vt_l, rho_r * vt_r
    flux_sum = jnp.stack((mn_l + mn_r,
                          mn_l * vn_l + mn_r * vn_r + cs**2 * (rho_l + rho_r),
                          mn_l * vt_l + mn_r * vt_r), axis=-1)
    speed = jnp.fmax(jnp.abs(vn_l) + cs, jnp.abs(vn_r) + cs)
    jump = jnp.stack((rho_r - rho_l, mn_r - mn_l, mt_r - mt_l), axis=-1)
    return 0.5 * (flux_sum - speed[..., None] * jump)


def _hlle_flux(rho_l, rho_r, vn_l, vn_r, vt_l, vt_r, cs):
    sqrt_l, sqrt_r = jnp.sqrt(rho_l), jnp.sqrt(rho_r)
    vn = (sqrt_l * vn_l + sqrt_r * vn_r) * (1.0 / (sqrt_l + sqrt_r))
    al = jnp.fmin(vn - cs, vn_l - cs)
    ar = jnp.fmax(vn + cs, vn_r + cs)
    bm = jnp.where(al < 0.0, al, -1e-20)
    bp = jnp.where(ar > 0.0, ar, 1e-20)

    # Evaluate the physical fluxes in the frames moving with the bounding waves.
    dl, dr = vn_l - bm, vn_r - bp
    fl = jnp.stack((rho_l * dl, rho_l * vn_l * dl + cs**2 * rho_l,
                    rho_l * vt_l * dl), axis=-1)
    fr = jnp.stack((rho_r * dr, rho_r * vn_r * dr + cs**2 * rho_r,
                    rho_r * vt_r * dr), axis=-1)
    weight = 0.5 * (bp + bm) / (bp - bm)
    return 0.5 * (fl + fr) + weight[..., None] * (fl - fr)


RIEMANN_SOLVERS = {"llf": _llf_flux, "hlle": _hlle_flux, "roe": _roe_flux}


def isothermal_flux(left, right, cs, axis, rsolver="roe"):
    """Return gas and passive scalar fluxes in x/y component order."""
    normal, tangent = (1, 2) if axis == 1 else (2, 1)
    states = (left[..., 0], right[..., 0], left[..., normal], right[..., normal],
              left[..., tangent], right[..., tangent])
    flux = RIEMANN_SOLVERS[rsolver](*states, cs)
    if axis == 0:
        flux = jnp.stack((flux[..., 0], flux[..., 2], flux[..., 1]), axis=-1)
    if left.shape[-1] > 3:
        mass_flux = flux[..., :1]
        upwind = jnp.where(mass_flux >= 0, left[..., 3:], right[..., 3:])
        flux = jnp.concatenate((flux, mass_flux * upwind), axis=-1)
    return flux


def roe_isothermal_flux(left, right, cs, axis):
    """Return AthenaK's isothermal Roe flux, including any passive scalars."""
    return isothermal_flux(left, right, cs, axis)


def face_velocity_gradients(W, dx, dy, axis):
    """Return (dv/dx, dv/dy) at each low face in the given direction."""
    velocity = W[..., 1:3]
    dn, dt = (dx, dy) if axis == 1 else (dy, dx)
    normal = (velocity - jnp.roll(velocity, 1, axis=axis)) / dn

    # Average the centered transverse difference from the two cells at the face.
    plus = jnp.roll(velocity, -1, axis=1 - axis)
    minus = jnp.roll(velocity, 1, axis=1 - axis)
    transverse = ((plus + jnp.roll(plus, 1, axis=axis))
                  - (minus + jnp.roll(minus, 1, axis=axis))) / (4.0 * dt)
    return (normal, transverse) if axis == 1 else (transverse, normal)


def smagorinsky_viscosity(dv_dx, dv_dy, dx, dy, smagorinsky_cs):
    """Kinematic eddy viscosity (Cs * sqrt(dx * dy))^2 times the strain norm."""
    shear = dv_dx[..., 1] + dv_dy[..., 0]
    strain_squared = 2.0 * (dv_dx[..., 0]**2 + dv_dy[..., 1]**2) + shear**2
    return smagorinsky_cs**2 * dx * dy * jnp.sqrt(strain_squared)


def viscous_momentum_flux(W, dx, dy, viscosity, smagorinsky_cs, axis):
    """Return the negative viscous stress in the x and y momentum fluxes."""
    dv_dx, dv_dy = face_velocity_gradients(W, dx, dy, axis)
    dvx_dx, dvy_dy = dv_dx[..., 0], dv_dy[..., 1]
    shear = dv_dx[..., 1] + dv_dy[..., 0]
    # AthenaK uses the three-dimensional trace subtraction in planar flow.
    if axis == 1:
        normal = (4.0 / 3.0) * dvx_dx - (2.0 / 3.0) * dvy_dy
        stress = jnp.stack((normal, shear), axis=-1)
    else:
        normal = (4.0 / 3.0) * dvy_dy - (2.0 / 3.0) * dvx_dx
        stress = jnp.stack((shear, normal), axis=-1)

    nu = viscosity
    if smagorinsky_cs > 0.0:
        nu += smagorinsky_viscosity(dv_dx, dv_dy, dx, dy, smagorinsky_cs)
    rho = W[..., 0]
    mu = 0.5 * nu * (rho + jnp.roll(rho, 1, axis=axis))
    return -mu[..., None] * stress


def stress_face_fluxes(tau):
    """Turn cell-centered (tau_xx, tau_xy, tau_yy) into momentum face fluxes."""
    if tau.ndim != 3 or tau.shape[-1] != 3:
        raise ValueError("Expected stress with shape (ny, nx, 3)")

    # Each face is indexed by the cell on its high-coordinate side.
    # Average the two adjacent stresses, wrapping across the periodic seam.
    tau_x = tau[..., :2]
    tau_y = tau[..., 1:]
    flux_x = 0.5 * (jnp.roll(tau_x, 1, axis=1) + tau_x)
    flux_y = 0.5 * (jnp.roll(tau_y, 1, axis=0) + tau_y)
    return flux_x, flux_y


def flux_divergence(flux_x, flux_y, dx, dy):
    """Difference the high and low faces of each periodic cell."""
    dflux_dx = (jnp.roll(flux_x, -1, axis=1) - flux_x) / dx
    dflux_dy = (jnp.roll(flux_y, -1, axis=0) - flux_y) / dy
    return dflux_dx + dflux_dy
