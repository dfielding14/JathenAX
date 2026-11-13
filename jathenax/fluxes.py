"""Face fluxes and their divergence on the periodic grid."""

import jax.numpy as jnp


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


def roe_isothermal_flux(left, right, cs, axis):
    """Return AthenaK's isothermal Roe flux and advect any passive scalars."""
    normal, tangent = (1, 2) if axis == 1 else (2, 1)
    rho_l, rho_r = left[..., 0], right[..., 0]
    vn_l, vn_r = left[..., normal], right[..., normal]
    vt_l, vt_r = left[..., tangent], right[..., tangent]
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
    flux = jnp.where(use_llf[..., None], llf, flux)
    if axis == 0:
        flux = jnp.stack((flux[..., 0], flux[..., 2], flux[..., 1]), axis=-1)
    if left.shape[-1] > 3:
        mass_flux = flux[..., :1]
        upwind = jnp.where(mass_flux >= 0, left[..., 3:], right[..., 3:])
        flux = jnp.concatenate((flux, mass_flux * upwind), axis=-1)
    return flux


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
