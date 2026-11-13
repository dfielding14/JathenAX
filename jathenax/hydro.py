"""Primitive recovery, Runge–Kutta stages, and timestep selection."""

import jax.numpy as jnp

from .forcing import normalize_force
from .fluxes import (
    face_velocity_gradients,
    flux_divergence,
    isothermal_flux,
    reconstruct_faces,
    smagorinsky_viscosity,
    viscous_momentum_flux,
)


# Each row is (gam0, gam1, beta, delta). The saved register starts as a
# copy of U at the beginning of each timestep. RK4 accumulates into it
# before later stages. None means the stage leaves that register alone.
RK_STAGES = {
    "rk2": (
        (0.0, 1.0, 1.0, None),
        (0.5, 0.5, 0.5, None),
    ),
    "rk3": (
        (0.0, 1.0, 1.0, None),
        (0.25, 0.75, 0.25, None),
        (2.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0, None),
    ),
    "rk4": (
        (0.0, 1.0, 1.193743905974738, None),
        (0.121098479554482, 0.721781678111411, 0.099279895495783, 0.217683334308543),
        (-3.843833699660025, 2.121209265338722, 1.131678018054042, 1.065841341361089),
        (0.546370891121863, 0.198653035682705, 0.310665766509336, 0.0),
    ),
}


def rk_stage(U, U_saved, div_flux, dt, coefficients):
    """Apply one AthenaK RK stage to the conserved variables and saved register."""
    gam0, gam1, beta, delta = coefficients
    if delta is not None:
        U_saved = U_saved + delta * U

    stage_dt = beta * dt
    if gam0 != 0.0 and gam0 + gam1 == 1.0:
        # Blend increments so equal states stay equal even in float32.
        U_next = U_saved + (gam0 * (U - U_saved) - stage_dt * div_flux)
    else:
        U_next = gam0 * U + gam1 * U_saved - stage_dt * div_flux
    return U_next, U_saved


def recover_isothermal(U, dfloor):
    """Recover density, velocity, and scalar concentrations."""
    rho = jnp.where(U[..., 0] < dfloor, dfloor, U[..., 0])
    U = jnp.concatenate((rho[..., None], U[..., 1:]), axis=-1)
    specific = (1.0 / rho)[..., None] * U[..., 1:]
    W = jnp.concatenate((rho[..., None], specific), axis=-1)
    return U, W


def isothermal_timestep(W, dx, dy, cs, cfl, previous_dt, time, tlim, *,
                        viscosity=0.0, smagorinsky_cs=0.0, drag_rate=0.0):
    """Apply AthenaK's hydro, diffusion, drag, growth, and end-time limits."""
    dt_x = jnp.min(dx / (jnp.abs(W[..., 1]) + cs))
    dt_y = jnp.min(dy / (jnp.abs(W[..., 2]) + cs))
    dt_limit = jnp.minimum(dt_x, dt_y)

    if smagorinsky_cs > 0.0:
        max_nu_sgs = 0.0
        for axis in (0, 1):
            dv_dx, dv_dy = face_velocity_gradients(W, dx, dy, axis)
            nu_sgs = smagorinsky_viscosity(dv_dx, dv_dy, dx, dy, smagorinsky_cs)
            max_nu_sgs = jnp.maximum(max_nu_sgs, jnp.max(nu_sgs))
        h = jnp.minimum(dx, dy)
        # The factor of two is the tangent viscosity for nonlinear shear.
        # With zero diffusion this bound is infinite.
        max_rate = (viscosity + 2.0 * max_nu_sgs) / (h * h)
        dt_limit = jnp.minimum(dt_limit, 0.1875 / max_rate)
    elif viscosity > 0.0:
        h = jnp.minimum(dx, dy)
        dt_limit = jnp.minimum(dt_limit, 0.25 * (h * h) / viscosity)

    if drag_rate > 0.0:
        dt_limit = jnp.minimum(dt_limit, 1.0 / drag_rate)

    dt = jnp.minimum(2.0 * previous_dt, cfl * dt_limit)
    return jnp.where((time < tlim) & (time + dt > tlim), tlim - time, dt)


def isothermal_step(U, dt, dx, dy, cs, dfloor, integrator, *,
                    viscosity=0.0, smagorinsky_cs=0.0, drag_rate=0.0,
                    raw_force=None, dedt=0.0, trace=False, reconstruct="ppm4", rsolver="roe"):
    """Advance one periodic isothermal step with optional scalar densities."""
    U, W = recover_isothermal(U, dfloor)
    U_saved = U
    stages = []
    schedule = RK_STAGES[integrator]
    for stage, coefficients in enumerate(schedule, 1):
        left_x, right_x = reconstruct_faces(W, 1, reconstruct, dfloor)
        left_y, right_y = reconstruct_faces(W, 0, reconstruct, dfloor)
        flux_x = isothermal_flux(left_x, right_x, cs, 1, rsolver)
        flux_y = isothermal_flux(left_y, right_y, cs, 0, rsolver)
        if viscosity > 0.0 or smagorinsky_cs > 0.0:
            visc_x = viscous_momentum_flux(W, dx, dy, viscosity, smagorinsky_cs, axis=1)
            visc_y = viscous_momentum_flux(W, dx, dy, viscosity, smagorinsky_cs, axis=0)
            flux_x = flux_x.at[..., 1:3].add(visc_x)
            flux_y = flux_y.at[..., 1:3].add(visc_y)
        div_flux = flux_divergence(flux_x, flux_y, dx, dy)
        U, U_saved = rk_stage(U, U_saved, div_flux, dt, coefficients)
        if drag_rate > 0.0:
            stage_dt = coefficients[2] * dt
            drag = (stage_dt * drag_rate * W[..., 0, None]) * W[..., 1:3]
            U = U.at[..., 1:3].add(-drag)

        # Normalize the kick against the conserved state after the final drag update.
        kick = raw_force is not None and stage == len(schedule)
        if kick:
            before_kick = U
            acceleration = normalize_force(U, raw_force, dt, dedt)
            U = U.at[..., 1:3].add(U[..., 0, None] * (acceleration * dt))

        # The AthenaK stage callback observes U before recovery and floors.
        if trace:
            stages.append(dict(
                primitives=W, left_x=left_x, right_x=right_x, left_y=left_y,
                right_y=right_y, flux_x=flux_x, flux_y=flux_y, saved=U_saved,
                conserved=U,
            ))
            if kick:
                stages[-1].update(before_kick=before_kick, acceleration=acceleration)
        U, W = recover_isothermal(U, dfloor)
    if trace:
        return U, tuple(stages)
    return U
