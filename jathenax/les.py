"""Periodic cubic Form A LES for the monatomic, stress-only pilot."""

from typing import NamedTuple

import jax.numpy as jnp

from .fluxes import hydro_flux, ppm4_face_states
from .hydro import RK_STAGES
from .tbnn import predict_stress, realizability_scale, stress_inputs


GAMMA = 5 / 3
RK3_WEIGHTS = (1/6, 1/6, 2/3)
STRESS_ROWS = ((0, 1, 2), (1, 3, 4), (2, 4, 5))


class Transport(NamedTuple):
    heat: float
    kinetic: float
    decay: float


def conserved(fields):
    """Drop the redundant thermal energy from a filtered reference state."""
    return jnp.concatenate((fields[..., :5], fields[..., 6:7]), axis=-1)


def filtered_state(q):
    """Recover (density, momentum, total energy, thermal energy, SGS energy)."""
    if q.ndim != 4 or q.shape[-1] != 6 or len(set(q.shape[:3])) != 1:
        raise ValueError("Form A requires a cubic 3D grid with six conserved fields")
    bulk = 0.5 * jnp.sum(q[..., 1:4]**2, axis=-1) / q[..., 0]
    thermal = q[..., 4] - bulk - q[..., 5]
    return jnp.concatenate((q[..., :5], thermal[..., None], q[..., 5:6]), axis=-1)


def face_average(a, axis):
    return 0.5 * (a + jnp.roll(a, 1, axis=axis))


def unit_transport(fields, k0, axis):
    """Low-face enthalpy and SGS-energy diffusion fluxes for unit coefficients."""
    rho = fields[..., 0]
    h, k = GAMMA * fields[..., 5] / rho, fields[..., 6] / rho
    vk = k / (jnp.sqrt(k + k0) + jnp.sqrt(k0))
    # Delta in the diffusivity cancels the gradient's cell width.
    mu = face_average(rho * vk, axis)
    return -mu * (h - jnp.roll(h, 1, axis)), -mu * (k - jnp.roll(k, 1, axis))


def analytic_stress(model, fields):
    inputs = stress_inputs(fields, GAMMA, model["k0"])
    tau = model["c_nu"] * inputs["unit_eddy"]
    return tau * realizability_scale(tau, inputs["K"])[..., None]


def kinetic_work(q, impulse):
    return jnp.sum((q[..., 1:4] + 0.5 * impulse) * impulse, axis=-1) / q[..., 0]


def apply_impulse(q, impulse, sgs_work):
    """A reference kick transfers momentum and SGS energy and preserves thermal energy."""
    work = kinetic_work(q, impulse) + sgs_work
    q = jnp.concatenate((q[..., :1], q[..., 1:4] + impulse,
                         q[..., 4:5] + work[..., None], q[..., 5:6] + sgs_work[..., None]), -1)
    return q, work


def les_timestep(q, dx, model, transport, tau_cool, cfl=0.4):
    """Pilot acoustic, analytic diffusion, exchange, and cooling step estimate."""
    fields = filtered_state(q)
    rho, K = fields[..., 0], fields[..., 6]
    sound = jnp.sqrt(GAMMA*(GAMMA-1)*(fields[..., 5]+K)/rho)
    velocity = fields[..., 1:4]/rho[..., None]
    acoustic = (jnp.abs(velocity).max(axis=-1) + sound).max() / dx
    speed = jnp.sqrt(jnp.maximum(K/rho, 0.)).max() / dx
    diffusion = 6 * jnp.maximum((4/3)*model["c_nu"], jnp.maximum(
        GAMMA*transport.heat, transport.kinetic)) * speed
    rate = jnp.maximum(jnp.maximum(acoustic, diffusion), jnp.maximum(transport.decay*speed, 1/tau_cool))
    return cfl/rate


def les_rhs(fields, tau, dx, model, transport):
    """Flux divergence including compatible production, before cooling and exchange."""
    rho, mom = fields[..., :1], fields[..., 1:4]
    v = mom / rho
    thermal, K = fields[..., 5:6], fields[..., 6:7]
    # At gamma=5/3 the effective Euler internal energy is U+K and pressure is 2(U+K)/3.
    w = jnp.concatenate((rho, v, thermal + K, K / rho), axis=-1)
    divergence = 0.
    for normal, row in enumerate(STRESS_ROWS):
        axis = 2 - normal
        left, right = ppm4_face_states(w, axis)
        flux = hydro_flux(left, right, axis, eos="ideal", gamma=GAMMA, cs=1., rsolver="roe")
        mass = flux[..., :1]
        vn = mass / jnp.where(mass >= 0, left[..., :1], right[..., :1])
        stress = face_average(jnp.stack([tau[..., i] for i in row], axis=-1), axis)
        velocity = face_average(v, axis).at[..., normal].set(vn[..., 0])
        work = (stress * velocity).sum(axis=-1, keepdims=True)
        heat, kinetic = unit_transport(fields, model["k0"], axis)
        heat = transport.heat * heat[..., None]
        kinetic = transport.kinetic * kinetic[..., None]
        flux = jnp.concatenate((mass, flux[..., 1:4] + stress,
                                flux[..., 4:5] + work + heat/GAMMA + kinetic,
                                flux[..., 5:6] + kinetic), axis=-1)

        def difference(a):
            return (jnp.roll(a, -1, axis=axis) - a) / dx

        production = ((2/3) * K * difference(vn) + difference(work)
                      - (v * difference(stress)).sum(axis=-1, keepdims=True))
        divergence = divergence + difference(flux).at[..., 5:6].add(production)
    return divergence


def les_step(q, dt, dx, model, transport, tau_cool, *, stress="tbnn", measured=None):
    """Take one unforced SSPRK3 step. Apply any kick separately at its physical time."""
    if stress not in ("analytic", "measured", "tbnn") or (stress == "measured" and measured is None):
        raise ValueError("select analytic, tbnn, or measured stress with three stage tensors")
    initial = q
    loss, minima = 0., jnp.full((3,), jnp.inf, q.dtype)
    # Explicit stages also avoid an erroneous transposed inner scan on JAX Metal.
    for index, ((a, _, beta, _), weight) in enumerate(zip(RK_STAGES["rk3"], RK3_WEIGHTS)):
        fields = filtered_state(q)
        if stress == "measured":
            tau = measured[index]
        elif stress == "tbnn":
            tau = predict_stress(model, fields)
        else:
            tau = analytic_stress(model, fields)
        div = les_rhs(fields, tau, dx, model, transport)
        q = initial + a * (q - initial) - beta * dt * div
        cooling = fields[..., 5] / tau_cool
        q = q.at[..., 4].add(-beta * dt * cooling)
        k = fields[..., 6] / fields[..., 0]
        speed = jnp.where(k > 0, jnp.sqrt(jnp.where(k > 0, k, 1.)), 0.)
        decay = beta * dt * transport.decay * fields[..., 6] * speed / dx
        q = q.at[..., 5].add(-jnp.minimum(decay, jnp.maximum(q[..., 5], 0.)))
        after = filtered_state(q)
        minima = jnp.minimum(minima, jnp.stack([after[..., i].min() for i in (0, 5, 6)]))
        loss = loss + dt * weight * cooling.mean()
    return q, loss, minima
