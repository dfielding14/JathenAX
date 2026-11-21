"""Benchmark-only equivalents for the 3D, five-variable ideal-gas pilot."""

import jax
import jax.numpy as jnp


def ppm_shared_edges(w, axis):
    """Construct each shared bounded face once, then limit it separately on each side."""
    n = w.shape[axis]
    def cut(a, start, size):
        return jax.lax.slice_in_dim(a, start, start + size, axis=axis)
    halo = jnp.concatenate((cut(w, n-3, 3), w, cut(w, 0, 2)), axis=axis)
    qm2, qm1, cell, qp1 = (cut(halo, k, n+2) for k in range(4))
    edge = (7.0 * (cell + qm1) - (qm2 + qp1)) / 12.0
    edge = jnp.fmin(jnp.fmax(edge, jnp.fmin(cell, qm1)), jnp.fmax(cell, qm1))
    low, high = cut(edge, 0, n+1), cut(edge, 1, n+1)
    center = cut(halo, 2, n+1)
    dlow, dhigh = low-center, high-center
    extreme = dlow*dhigh >= 0
    low = jnp.where(jnp.abs(dlow) >= 2*jnp.abs(dhigh), center - 2*dhigh, low)
    high = jnp.where(jnp.abs(dhigh) >= 2*jnp.abs(dlow), center - 2*dlow, high)
    return cut(jnp.where(extreme, center, high), 0, n), cut(jnp.where(extreme, center, low), 1, n)


def scalar_roe(left, right, axis, *, eos, gamma, cs, rsolver):
    """Keep scalars separate until returning the final five-component Roe flux."""
    assert left.ndim == 4 and left.shape[-1] == 5 and eos == "ideal" and rsolver == "roe"
    normal = 2-axis
    def side(w):
        rho, e = w[...,0], w[...,4]
        vel = tuple(w[...,i+1] for i in range(3))
        vn = vel[normal]
        p = (gamma-1)*e
        mom = tuple(rho*v for v in vel)
        energy = e + .5*sum(m*v for m,v in zip(mom,vel))
        mass_flux = rho*vn
        momentum_flux = tuple(mass_flux*v + (p if i == normal else 0.) for i,v in enumerate(vel))
        return (rho,*mom,energy), (mass_flux,*momentum_flux,(energy+p)*vn), vel, p, jnp.sqrt(gamma*p/rho)

    ul, fl, vl, pl, cl = side(left)
    ur, fr, vr, pr, cr = side(right)
    rl, rr = jnp.sqrt(ul[0]), jnp.sqrt(ur[0])
    inv = 1/(rl+rr)
    vel = tuple((rl*a + rr*b)*inv for a,b in zip(vl,vr))
    wl, wr = rl*inv, rr*inv
    variance = wl*wr*sum((b-a)**2 for a,b in zip(vl,vr))
    sound = jnp.sqrt(wl*cl**2 + wr*cr**2 + .5*(gamma-1.)*variance)
    vn = vel[normal]
    minus, plus = vn-sound, vn+sound
    dr = (pr-pl)/sound**2
    dv = rl*rr*(vr[normal]-vl[normal])/sound
    am, ap = .5*(dr-dv), .5*(dr+dv)
    adv = jnp.abs(vn)
    cm, cp = (jnp.abs(minus)-adv)*am, (jnp.abs(plus)-adv)*ap
    acoustic, normal_term = cm+cp, sound*(cp-cm)
    corr_m = tuple(v*acoustic + (normal_term if i == normal else 0.) for i,v in enumerate(vel))
    enthalpy = sound**2/(gamma-1.) + .5*sum(v**2 for v in vel)
    corr = (acoustic, *corr_m, enthalpy*acoustic + vn*normal_term)
    llf_speed = jnp.maximum(jnp.abs(vl[normal])+cl, jnp.abs(vr[normal])+cr)
    star = ul[0] + am
    fallback = (star < 0) | (star + (ur[0]-ul[0]-dr) < 0)
    result = []
    for l, r, a, b, c in zip(fl, fr, ul, ur, corr):
        jump = b-a
        f = .5*(l+r-adv*jump-c)
        f = jnp.where(minus >= 0, l, jnp.where(plus <= 0, r, f))
        result.append(jnp.where(fallback, .5*(l+r-llf_speed*jump), f))
    return jnp.stack(result, axis=-1)


def yz_basis(basis):
    """Fixed transverse phases shared by all updates at this resolution."""
    _, _, cy, sy, cz, sz = basis
    cy, sy = cy[:,None,:], sy[:,None,:]
    cz, sz = cz[:,:,None], sz[:,:,None]
    c, s = cz*cy-sz*sy, cz*sy+sz*cy
    return jnp.concatenate((c,-s),axis=0).reshape(2*len(c),-1).T


def render_matmul(amplitudes, basis, yz):
    cx, sx = basis[0][...,None], basis[1][...,None]
    a, b = amplitudes[:,None,:,0], amplitudes[:,None,:,1]
    x = jnp.concatenate((cx*a-sx*b, sx*a+cx*b),axis=0)
    planes = jnp.matmul(yz, x.reshape(len(x),-1), precision=jax.lax.Precision.HIGHEST)
    return planes.reshape(basis[4].shape[1], basis[2].shape[1], cx.shape[1], amplitudes.shape[1])
