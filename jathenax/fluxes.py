"""Face fluxes and their divergence on the periodic grid."""

from math import prod

from jax import custom_jvp, lax
import jax.numpy as jnp


def dc_face_states(W, axis):
    """Use the two adjacent cell averages at each face."""
    return jnp.roll(W, 1, axis=axis), W


def _periodic_stencil(W, axis, before, after, width):
    """Overlapping views of a single periodic halo, including the seam face."""
    n = W.shape[axis]
    if n < max(before, after):
        pads = [(0, 0)] * W.ndim
        pads[axis] = (before, after)
        halo = jnp.pad(W, pads, mode="wrap")
    else:
        halo = jnp.concatenate((lax.slice_in_dim(W, n - before, n, axis=axis), W,
                                lax.slice_in_dim(W, 0, after, axis=axis)), axis=axis)
    length = n + before + after - width + 1
    return tuple(lax.slice_in_dim(halo, i, i + length, axis=axis) for i in range(width))


def _face_pair(high, low, axis):
    """Pair the high edge of cell i-1 with the low edge of cell i."""
    n = high.shape[axis] - 1
    return lax.slice_in_dim(high, 0, n, axis=axis), lax.slice_in_dim(low, 1, n + 1, axis=axis)


def _rounded(x):
    """Preserve an AthenaK intermediate's rounding across XLA multiply-add fusion."""
    return jnp.copysign(jnp.abs(x), x)


def _multiply_add(a, b, c):
    """Preserve the reference FP64 contraction order without a batched dot in FP32."""
    if jnp.result_type(a, b, c) != jnp.float64:
        return a*b + c
    a, b, c = jnp.broadcast_arrays(a, b, c)
    return jnp.einsum("...i,...i->...", jnp.stack((c, a), axis=-1),
                      jnp.stack((jnp.ones_like(b), b), axis=-1), precision=lax.Precision.HIGHEST)


def plm_face_states(W, axis):
    """Reconstruct a piecewise linear profile with AthenaK's harmonic limiter."""
    qm2, qm1, cell, qp1 = _periodic_stencil(W, axis, 2, 1, 4)
    jump = cell - qm1

    def half_slope(a, b):
        product = a * b
        monotone = product > 0
        return jnp.where(monotone, product / jnp.where(monotone, a + b, 1.0), 0.0)

    return qm1 + half_slope(qm1 - qm2, jump), cell - half_slope(jump, qp1 - cell)


def ppm4_face_states(W, axis):
    """Return left and right primitives at the low face of each cell."""
    qm2, qm1, cell, qp1 = _periodic_stencil(W, axis, 3, 2, 4)
    # Preserve the reference division at nearly stationary HLLC contacts.
    divisor = lax.optimization_barrier(jnp.asarray(12.0, W.dtype))
    edge = (7.0 * (cell + qm1) - (qm2 + qp1)) / divisor
    edge = jnp.fmin(jnp.fmax(edge, jnp.fmin(cell, qm1)), jnp.fmax(cell, qm1))
    # Neighboring cells share this bounded edge, then limit their own parabola.
    low, high = _face_pair(edge, edge, axis)
    W = lax.slice_in_dim(cell, 0, cell.shape[axis] - 1, axis=axis)

    # Flatten extrema and limit asymmetric parabolas using AthenaK's PPM4 rules.
    dhigh = high - W
    dlow = low - W
    extremum = dhigh * dlow >= 0.0
    low = jnp.where(jnp.abs(dlow) >= 2.0 * jnp.abs(dhigh), W - 2.0 * dhigh, low)
    high = jnp.where(jnp.abs(dhigh) >= 2.0 * jnp.abs(dlow), W - 2.0 * dlow, high)
    low = jnp.where(extremum, W, low)
    high = jnp.where(extremum, W, high)
    return _face_pair(high, low, axis)


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
    a0, a1, a2 = 0.1 * w0, 0.6 * w1, 0.3 * w2
    f0 = _multiply_add(11.0, q, 2*qm2 - _rounded(7*qm1))
    f1 = _rounded(5*q) - qm1 + 2*qp1
    f2 = 2*q + _rounded(5*qp1) - qp2
    return (f0 * a0 + f1 * a1 + f2 * a2) / (6.0 * (a0 + a1 + a2))


@custom_jvp
def _ratio(numerator, denominator):
    return numerator / denominator


@_ratio.defjvp
def _ratio_jvp(primals, tangents):
    numerator, denominator = primals
    dnumerator, ddenominator = tangents
    ratio = numerator / denominator
    # The squared WENO or HLLC denominator can underflow in the FP32 quotient rule.
    return ratio, (dnumerator - ratio * ddenominator) / denominator


def _wenoz_weights32(W, terms):
    eps = float(W.dtype.type(1e-42))
    factor = 2.0**20

    def scaled(terms):
        # The same power of two scales every beta and epsilon into normal FP32 range.
        beta = tuple((13/12)*(a*factor)**2 + 0.25*(b*factor)**2 for a, b in terms)
        tau = abs(beta[0] - beta[2])
        den = tuple(b + eps*factor**2 for b in beta)
        dmin = lax.stop_gradient(jnp.minimum(den[0], jnp.minimum(den[1], den[2])))
        inv = lax.stop_gradient(1.0 / jnp.maximum(tau, dmin))
        base, contrast = (dmin*inv)**2, tau*inv
        return tuple(base + (contrast*_ratio(dmin, d))**2 for d in den)

    def wide(terms):
        # For large fields, form sqrt(beta + eps) without overflowing the squared differences.
        def root_indicator(a, b):
            scale = lax.stop_gradient(jnp.maximum(jnp.maximum(abs(a), abs(b)), eps**0.5))
            return scale * jnp.sqrt((13/12)*(a/scale)**2 + 0.25*(b/scale)**2 + (eps**0.5/scale)**2)

        h0, h1, h2 = (root_indicator(a, b) for a, b in terms)
        hmin = lax.stop_gradient(jnp.minimum(h0, jnp.minimum(h1, h2)))
        diff = abs(h0 - h2)
        root_tau = jnp.where(diff > 0, jnp.sqrt(jnp.where(diff > 0, diff, 1.0))*jnp.sqrt(h0+h2), 0.0)
        scale = lax.stop_gradient(jnp.maximum(hmin, root_tau))
        base, contrast = (hmin/scale)**4, (root_tau/scale)**2
        return tuple(base + (contrast*_ratio(hmin, h)**2)**2 for h in (h0, h1, h2))

    # Bound the scaled indicators so their reciprocals also stay normal.
    limit = (1.0 / float(jnp.finfo(W.dtype).tiny))**0.5 / (8*factor)
    return lax.cond(jnp.max(abs(W)) < limit, scaled, wide, terms)


def wenoz_face_states(W, axis):
    """Reconstruct both edges with AthenaK's fifth-order WENO-Z stencil."""
    qm2, qm1, q, qp1, qp2 = _periodic_stencil(W, axis, 3, 2, 5)
    terms = ((qm2 + q - 2*qm1, qm2 + 3*q - 4*qm1),
             (qm1 + qp1 - 2*q, qm1 - qp1),
             (qp2 + q - 2*qp1, qp2 + 3*q - 4*qp1))
    if W.dtype == jnp.float32:
        w0, w1, w2 = _wenoz_weights32(W, terms)
    else:
        b0, b1, b2 = (_multiply_add(13/12, a**2, 0.25*b**2) for a, b in terms)
        tau = abs(b0 - b2)
        w0, w1, w2 = (1.0 + _rounded((tau/(b + 1e-42))**2) for b in (b0, b1, b2))
    high = _wenoz_edge(qm2, qm1, q, qp1, qp2, w0, w1, w2)
    low = _wenoz_edge(qp2, qp1, q, qm1, qm2, w2, w1, w0)
    return _face_pair(high, low, axis)


RECONSTRUCTIONS = {
    "dc": dc_face_states, "plm": plm_face_states, "ppm4": ppm4_face_states,
    "ppmx": ppmx_face_states, "wenoz": wenoz_face_states,
}


def reconstruct_faces(W, axis, reconstruct, dfloor, *, energy_index=None, efloor=None):
    left, right = RECONSTRUCTIONS[reconstruct](W, axis)
    if reconstruct in ("ppmx", "wenoz"):
        def apply_floors(face):
            rho = jnp.fmax(face[..., :1], dfloor)
            if isinstance(energy_index, int) and 0 < energy_index < face.shape[-1]:
                energy = jnp.fmax(face[..., energy_index:energy_index + 1], efloor)
                fields = (rho, face[..., 1:energy_index], energy, face[..., energy_index + 1:])
                return jnp.concatenate(fields, axis=-1)
            face = jnp.concatenate((rho, face[..., 1:]), axis=-1)
            if energy_index is not None:
                face = face.at[..., energy_index].set(jnp.fmax(face[..., energy_index], efloor))
            return face

        left, right = apply_floors(left), apply_floors(right)
    return left, right


RIEMANN_SOLVERS = ("llf", "hlle", "hllc", "roe")


def _unit_normal(ndim, normal, dtype):
    # Materialize +1 on the host: Metal can lower a vector boolean-to-float cast to -1.
    return jnp.asarray(tuple(1.0 if i == normal else 0.0 for i in range(ndim)), dtype=dtype)


def _hll_flux(left, right, normal, unit_normal, eos, gamma, cs, rsolver):
    """AthenaK's moving-frame side fluxes and HLLE or HLLC wave weights."""
    ndim = left.ndim - 1
    ideal = eos == "ideal"
    if rsolver == "hllc" and not ideal:
        raise ValueError("HLLC requires an ideal-gas EOS")
    if left.dtype == jnp.float32:
        if ideal:
            gamma = jnp.asarray(gamma, dtype=left.dtype)
        else:
            cs = jnp.asarray(cs, dtype=left.dtype)

    def speed_squared(v):
        # AthenaK sums in cyclic order, starting with the normal component.
        axes = (*range(normal, ndim), *range(normal))
        return sum(v[..., i:i+1]**2 for i in axes)

    def state(w):
        rho, v = w[..., :1], w[..., 1:1+ndim]
        vn = v[..., normal:normal+1]
        if ideal:
            p = (gamma-1.0)*w[..., ndim+1:ndim+2]
            # Round the EOS pressure before either side forms its contact momentum flux.
            # Fusing that multiply into the flux can change the sign of a stationary contact.
            if rsolver == "hllc":
                p = _rounded(p)
            E = p*(1.0/(gamma-1.0)) + 0.5*rho*speed_squared(v)
            return rho, v, vn, p, jnp.sqrt(gamma*p/rho), E
        return rho, v, vn, cs**2*rho, cs, None

    rho_l, v_l, vn_l, p_l, c_l, E_l = state(left)
    rho_r, v_r, vn_r, p_r, c_r, E_r = state(right)
    if rsolver == "hlle":
        root_l, root_r = jnp.sqrt(rho_l), jnp.sqrt(rho_r)
        inv_sum = 1.0/(root_l + root_r)
        v_roe = (root_l*v_l + root_r*v_r)*inv_sum
        sound = cs
        if ideal:
            h_roe = ((E_l+p_l)/root_l + (E_r+p_r)/root_r)*inv_sum
            thermal = h_roe - 0.5*speed_squared(v_roe)
            zero_sound = thermal <= 0.0
            sound = jnp.where(zero_sound, 0.0,
                              jnp.sqrt((gamma-1.0)*jnp.where(zero_sound, 1.0, thermal)))
        vn_roe = v_roe[..., normal:normal+1]
        s_l = jnp.fmin(vn_roe-sound, vn_l-c_l)
        s_r = jnp.fmax(vn_roe+sound, vn_r+c_r)
    else:
        p_mid = 0.5*(p_l+p_r + (vn_l-vn_r)*(0.25*(rho_l+rho_r)*(c_l+c_r)))
        alpha = (gamma+1.0)/(2.0*gamma)

        def shock_factor(p):
            return jnp.sqrt(1.0 + alpha*jnp.where(p_mid <= p, 0.0, p_mid/p-1.0))

        s_l = vn_l - c_l*shock_factor(p_l)
        s_r = vn_r + c_r*shock_factor(p_r)

    lo = jnp.where(s_l < 0, s_l, -1e-20)
    hi = jnp.where(s_r > 0, s_r, 1e-20)

    def side_flux(rho, v, vn, p, E, wave):
        mass = rho*(vn-wave)
        fields = (mass, mass*v + p*unit_normal)
        if ideal:
            fields += (E*(vn-wave) + p*vn,)
        return jnp.concatenate(fields, axis=-1)

    fl = side_flux(rho_l, v_l, vn_l, p_l, E_l, lo)
    fr = side_flux(rho_r, v_r, vn_r, p_r, E_r, hi)
    if rsolver == "hlle":
        shift = 0.5*(hi+lo)/(hi-lo)
        return 0.5*(fl+fr) + shift*(fl-fr)

    ml, signed_r = rho_l*(vn_l-s_l), rho_r*(vn_r-s_r)
    tl, tr = p_l + ml*vn_l, p_r + signed_r*vn_r
    mr = -signed_r
    contact = (tl-tr)/(ml+mr)
    p_star = (ml*tr + mr*tl)/(ml+mr)
    p_star = jnp.where(p_star > 0, p_star, 0.0)
    from_left = contact >= 0
    gap = jnp.where(from_left, contact-lo, hi-contact)
    weight = _ratio(jnp.where(from_left, contact, -contact), gap)
    contact_pressure = _ratio(jnp.where(from_left, -lo, hi), gap)*p_star
    star = jnp.concatenate((jnp.zeros_like(contact), contact_pressure*unit_normal,
                            contact_pressure*contact), axis=-1)
    return weight*jnp.where(from_left, fl, fr) + star


def _roe_llf_flux(left, right, normal, unit_normal, eos, gamma, cs, rsolver):
    ndim = left.ndim - 1
    rho_l, rho_r = left[..., :1], right[..., :1]
    v_l, v_r = left[..., 1:1 + ndim], right[..., 1:1 + ndim]
    vn_l, vn_r = v_l[..., normal:normal+1], v_r[..., normal:normal+1]
    ideal = eos == "ideal"
    if ideal:
        e_l, e_r = left[..., ndim+1:ndim+2], right[..., ndim+1:ndim+2]
        p_l, p_r = (gamma-1.0)*e_l, (gamma-1.0)*e_r
        csq_l, csq_r = gamma*p_l/rho_l, gamma*p_r/rho_r
        c_l, c_r = jnp.sqrt(csq_l), jnp.sqrt(csq_r)
        E_l = e_l + 0.5*jnp.sum((rho_l*v_l)*v_l, axis=-1, keepdims=True)
        E_r = e_r + 0.5*jnp.sum((rho_r*v_r)*v_r, axis=-1, keepdims=True)
    else:
        p_l, p_r = cs**2 * rho_l, cs**2 * rho_r
        c_l = c_r = cs

    if rsolver == "roe":
        root_l, root_r = jnp.sqrt(rho_l), jnp.sqrt(rho_r)
        inv_sum = 1.0 / (root_l + root_r)
        velocity = (root_l*v_l + root_r*v_r) * inv_sum
        vn = velocity[..., normal:normal+1]
        sound = cs
        if ideal:
            # Thermal terms plus velocity variance avoid high-Mach cancellation.
            wl, wr = root_l * inv_sum, root_r * inv_sum
            variance = wl*wr * jnp.sum((v_r-v_l)**2, axis=-1, keepdims=True)
            sound = jnp.sqrt(wl*csq_l + wr*csq_r + 0.5*(gamma-1.0)*variance)
        minus, plus = vn - sound, vn + sound

    if rsolver in ("roe", "llf"):
        speed = jnp.maximum(jnp.abs(vn_l)+c_l, jnp.abs(vn_r)+c_r)
        if rsolver == "roe":
            density_wave = (p_r-p_l) / sound**2
            speed_wave = root_l*root_r * (vn_r-vn_l) / sound
            a_minus = 0.5*(density_wave-speed_wave)
            a_plus = 0.5*(density_wave+speed_wave)
            star = rho_l + a_minus
            middle_jump = rho_r-rho_l-density_wave if ideal else a_plus
            fallback = (star < 0) | (star + middle_jump < 0)
            use_l, use_r = (minus >= 0) & ~fallback, (plus <= 0) & ~fallback
            weight_l = jnp.where(use_l, 1.0, jnp.where(use_r, 0.0, 0.5))
            weight_r = 1.0-weight_l
            adv = jnp.abs(vn)
            diffusion = jnp.where(use_l | use_r, 0.0, 0.5*jnp.where(fallback, speed, adv))
            cm, cp = (jnp.abs(minus)-adv)*a_minus, (jnp.abs(plus)-adv)*a_plus
            use_roe = ~(use_l | use_r | fallback)
            acoustic = jnp.where(use_roe, -0.5*(cm+cp), 0.0)
            normal_term = jnp.where(use_roe, -0.5*sound*(cp-cm), 0.0)
        else:
            weight_l = weight_r = 0.5
            diffusion = 0.5*speed
        transport_l = weight_l*vn_l + diffusion
        transport_r = weight_r*vn_r - diffusion
    else:
        raise ValueError(f"unknown Riemann solver {rsolver!r}")

    pressure = weight_l * p_l + weight_r * p_r
    if ideal:
        pressure_work = weight_l*p_l*vn_l + weight_r*p_r*vn_r

    # Only the final flux is assembled. Each field uses the same advective weights.
    mass_l, mass_r = transport_l*rho_l, transport_r*rho_r
    mass = mass_l + mass_r
    momentum = mass_l*v_l + mass_r*v_r + pressure*unit_normal
    if ideal:
        energy = transport_l*E_l + transport_r*E_r + pressure_work
    if rsolver == "roe":
        mass = mass + acoustic
        momentum = momentum + acoustic*velocity + normal_term*unit_normal
        if ideal:
            enthalpy = sound**2/(gamma-1.0) + 0.5*jnp.sum(velocity**2, axis=-1, keepdims=True)
            energy = energy + enthalpy*acoustic + vn*normal_term
    return jnp.concatenate((mass, momentum, energy) if ideal else (mass, momentum), axis=-1)


def hydro_flux(left, right, axis, *, eos, gamma, cs, rsolver):
    """Euler flux in physical component order at the low face on an array axis."""
    left, right = jnp.asarray(left), jnp.asarray(right)
    ndim = left.ndim - 1
    normal = ndim - 1 - axis
    unit_normal = _unit_normal(ndim, normal, left.dtype)
    solver = _hll_flux if rsolver in ("hlle", "hllc") else _roe_llf_flux
    flux = solver(left, right, normal, unit_normal, eos, gamma, cs, rsolver)
    ngas = flux.shape[-1]
    if left.shape[-1] > ngas:
        mass = flux[..., :1]
        upwind = jnp.where(mass >= 0, left[..., ngas:], right[..., ngas:])
        flux = jnp.concatenate((flux, mass * upwind), axis=-1)
    return flux


def isothermal_flux(left, right, cs, axis, rsolver="roe"):
    return hydro_flux(left, right, axis, eos="isothermal", gamma=None, cs=cs, rsolver=rsolver)


def roe_isothermal_flux(left, right, cs, axis):
    return isothermal_flux(left, right, cs, axis)


def velocity_gradients(W, spacing, normal):
    """Face derivatives grad[..., i, j] = d(v_i)/d(x_j) in physical axis order."""
    ndim = len(spacing)
    axis = ndim - 1 - normal
    velocity = W[..., 1:1 + ndim]
    columns = []
    for j, h in enumerate(spacing):
        if j == normal:
            derivative = (velocity - jnp.roll(velocity, 1, axis=axis)) / h
        else:
            transverse = ndim - 1 - j
            plus = jnp.roll(velocity, -1, axis=transverse)
            minus = jnp.roll(velocity, 1, axis=transverse)
            derivative = ((plus + jnp.roll(plus, 1, axis=axis))
                          - (minus + jnp.roll(minus, 1, axis=axis))) / (4.0 * h)
        columns.append(derivative)
    return jnp.stack(columns, axis=-1)


def eddy_viscosity(grad, spacing, smagorinsky_cs):
    strain = 0.5 * (grad + grad.swapaxes(-1, -2))
    delta_sq = prod(spacing)**(2.0 / len(spacing))
    return smagorinsky_cs**2 * delta_sq * jnp.sqrt(2.0 * jnp.sum(strain**2, axis=(-2, -1)))


def viscous_flux(W, spacing, viscosity, smagorinsky_cs, normal):
    """Negative Newtonian face stress with the physical three-dimensional trace."""
    grad = velocity_gradients(W, spacing, normal)
    stress = grad[..., :, normal] + grad[..., normal, :]
    divergence = jnp.trace(grad, axis1=-2, axis2=-1)
    unit_normal = _unit_normal(len(spacing), normal, W.dtype)
    stress = stress - (2.0 / 3.0) * divergence[..., None] * unit_normal
    if smagorinsky_cs > 0:
        viscosity = viscosity + eddy_viscosity(grad, spacing, smagorinsky_cs)
    rho = W[..., 0]
    axis = len(spacing) - 1 - normal
    mu = 0.5 * viscosity * (rho + jnp.roll(rho, 1, axis=axis))
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
