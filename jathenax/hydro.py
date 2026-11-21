"""Primitive recovery, Runge–Kutta stages, and timestep selection."""

from jax import lax
import jax.numpy as jnp

from .forcing import normalize_force
from .fluxes import (
    _multiply_add, _rounded, eddy_viscosity, hydro_flux, reconstruct_faces, stress_face_fluxes,
    velocity_gradients, viscous_flux,
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
    if gam0 != 0.0 and jnp.result_type(U) == jnp.float64:
        blend = _multiply_add(gam0, U, _rounded(gam1 * U_saved))
        U_next = _multiply_add(-stage_dt, div_flux, blend)
    elif gam0 != 0.0 and gam0 + gam1 == 1.0:
        # Blend increments so equal states stay equal even in float32.
        U_next = (U_saved - stage_dt * div_flux) + gam0 * (U - U_saved)
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


def recover_primitives(U, eos, gamma, dfloor, pfloor):
    """Apply floors and recover velocity, internal energy density, and concentrations."""
    ndim = U.ndim - 1
    if eos not in ("isothermal", "ideal"):
        raise ValueError("eos must be isothermal or ideal")
    if not 1 <= ndim <= 3 or U.shape[-1] < 1 + ndim + (eos == "ideal"):
        raise ValueError("state must have one to three spatial axes and all gas components")
    if eos == "isothermal":
        return recover_isothermal(U, dfloor)
    energy = ndim + 1
    rho = jnp.maximum(U[..., :1], dfloor)
    momentum = U[..., 1:energy]
    inv_rho = 1.0 / rho
    velocity = inv_rho * momentum
    # A one-ulp pressure change can select the other side of a nearly stationary HLLC contact.
    mx = momentum[..., :1]
    mom_squared = mx**2
    if ndim >= 2:
        mom_squared = _multiply_add(mx, mx, momentum[..., 1:2]**2)
    if ndim == 3:
        mz = momentum[..., 2:3]
        mom_squared = _multiply_add(mz, mz, mom_squared)
    kinetic = _rounded(0.5 * inv_rho * mom_squared)
    total = U[..., energy:energy + 1]
    internal = total - kinetic
    minimum = pfloor / (gamma - 1.0)
    # Preserve the original conserved energy exactly in cells above the floor.
    total = jnp.where(internal < minimum, kinetic + minimum, total)
    tracers = U[..., energy + 1:]
    U = jnp.concatenate((rho, momentum, total, tracers), axis=-1)
    W = jnp.concatenate((rho, velocity, jnp.maximum(internal, minimum), tracers / rho), axis=-1)
    return U, W


def hydro_timestep(W, spacing, cfl, previous_dt, time, tlim, *, eos="ideal",
                   gamma=5/3, cs=1.0, viscosity=0.0, smagorinsky_cs=0.0,
                   drag_rate=0.0, tau_cool=0.0):
    """Limit the timestep by the active directions, transport, sources and end time."""
    ndim = W.ndim - 1
    if len(spacing) != ndim:
        raise ValueError("supply one cell width per spatial axis in x, y, z order")
    sound = jnp.sqrt(gamma * (gamma - 1.0) * W[..., ndim + 1] / W[..., 0]) if eos == "ideal" else cs
    dt_limit = jnp.inf
    for i, h in enumerate(spacing):
        dt_limit = jnp.minimum(dt_limit, jnp.min(h / (jnp.abs(W[..., i + 1]) + sound)))
    if viscosity > 0.0 or smagorinsky_cs > 0.0:
        h_min = jnp.min(jnp.asarray(spacing))
        if smagorinsky_cs > 0.0:
            max_sgs = 0.0
            for normal in range(ndim):
                grad = velocity_gradients(W, spacing, normal)
                max_sgs = jnp.maximum(max_sgs, jnp.max(eddy_viscosity(grad, spacing, smagorinsky_cs)))
            # Bound the nonlinear tangent viscosity as in the planar solver.
            diffusive_dt = (0.375 / ndim) * h_min**2 / (viscosity + 2.0 * max_sgs)
        else:
            diffusive_dt = h_min**2 / (2.0 * ndim * viscosity)
        dt_limit = jnp.minimum(dt_limit, diffusive_dt)
    if drag_rate > 0.0:
        dt_limit = jnp.minimum(dt_limit, 1.0 / drag_rate)
    if tau_cool > 0.0:
        dt_limit = jnp.minimum(dt_limit, tau_cool)
    dt = jnp.minimum(2.0 * previous_dt, cfl * dt_limit)
    return jnp.where((time < tlim) & (time + dt > tlim), tlim - time, dt)


def hydro_step(U, dt, spacing, *, eos="ideal", gamma=5/3, cs=1.0, dfloor=1e-10,
               pfloor=1e-10, integrator="rk3", viscosity=0.0, smagorinsky_cs=0.0,
               drag_rate=0.0, tau_cool=0.0, cooling_ramp_time=0.0, cooling_ramp_exponent=1.0, time=0.0,
               raw_force=None, dedt=0.0, trace=False,
               reconstruct="ppm4", rsolver="roe", predict_stress=None, sgs_weights=None):
    """Advance one unsplit periodic step. Spacings are in physical x, y, z order."""
    ndim = U.ndim - 1
    if len(spacing) != ndim:
        raise ValueError("supply one cell width per spatial axis in x, y, z order")
    if tau_cool < 0 or (tau_cool > 0 and eos != "ideal"):
        raise ValueError("tau_cool must be nonnegative and cooling requires ideal gas")
    if not 0.0 <= cooling_ramp_time < float("inf"):
        raise ValueError("cooling_ramp_time must be finite and nonnegative")
    if not 0.5 < cooling_ramp_exponent < float("inf"):
        raise ValueError("a smooth cooling ramp requires a finite exponent greater than 0.5")
    if predict_stress is not None and (ndim != 2 or eos != "isothermal"):
        raise ValueError("the learned stress interface requires 2D isothermal hydro")
    if raw_force is not None and raw_force.shape != (*U.shape[:-1], ndim):
        raise ValueError("raw_force must have one acceleration per active direction and cell")

    U, W = recover_primitives(U, eos, gamma, dfloor, pfloor)
    U_saved = U
    momentum = slice(1, ndim + 1)
    energy = ndim + 1 if eos == "ideal" else None
    efloor = pfloor / (gamma - 1.0) if eos == "ideal" else None
    stages = []
    schedule = RK_STAGES[integrator]
    stage_offset = saved_offset = 0.0
    for stage, coefficients in enumerate(schedule, 1):
        if predict_stress is not None:
            primitives = W[..., :3]
            tau = jnp.asarray(predict_stress(sgs_weights, primitives))
            if tau.shape != primitives.shape or not jnp.issubdtype(tau.dtype, jnp.floating):
                raise ValueError("predict_stress must return a floating array of shape (ny, nx, 3)")
            stress = stress_face_fluxes(tau.astype(U.dtype))
        record = {}
        div_flux = 0.0
        for normal, h in enumerate(spacing):
            axis = ndim - 1 - normal
            left, right = reconstruct_faces(W, axis, reconstruct, dfloor,
                                            energy_index=energy, efloor=efloor)
            flux = hydro_flux(left, right, axis, eos=eos, gamma=gamma, cs=cs, rsolver=rsolver)
            if viscosity > 0.0 or smagorinsky_cs > 0.0:
                visc = viscous_flux(W, spacing, viscosity, smagorinsky_cs, normal)
                flux = flux.at[..., momentum].add(visc)
                if eos == "ideal":
                    v = W[..., momentum]
                    v_face = 0.5 * (v + jnp.roll(v, 1, axis=axis))
                    flux = flux.at[..., energy].add(jnp.sum(v_face * visc, axis=-1))
            if predict_stress is not None:
                flux = flux.at[..., momentum].add(stress[normal])
            # A constant spacing would otherwise become a rounded reciprocal in XLA.
            h = lax.optimization_barrier(jnp.asarray(h, U.dtype))
            div_flux = div_flux + (jnp.roll(flux, -1, axis=axis) - flux) / h
            if trace:
                name = "xyz"[normal]
                record.update({f"left_{name}": left, f"right_{name}": right, f"flux_{name}": flux})
        U, U_saved = rk_stage(U, U_saved, div_flux, dt, coefficients)
        stage_dt = coefficients[2] * dt
        if drag_rate > 0.0:
            drag = (stage_dt * drag_rate * W[..., :1]) * W[..., momentum]
            U = U.at[..., momentum].add(-drag)
            if eos == "ideal":
                U = U.at[..., energy].add(-jnp.sum(drag * W[..., momentum], axis=-1))
        if tau_cool > 0.0:
            strength = 1.0
            if cooling_ramp_time > 0.0:
                x = jnp.clip((time + stage_offset * dt) / cooling_ramp_time, 0.0, 1.0)
                strength = (x * x * (3.0 - 2.0 * x)) ** cooling_ramp_exponent
                # Advance t'=1 in the same RK registers to get the next stage's time in units of dt.
                stage_offset, saved_offset = rk_stage(stage_offset, saved_offset, -1.0, 1.0, coefficients)
            U = U.at[..., energy].add(-(stage_dt / tau_cool) * strength * W[..., energy])
            if trace:
                record["cooling_fraction"] = strength

        if raw_force is not None and stage == len(schedule):
            acceleration, scale, mean = normalize_force(U, raw_force, dt, dedt, return_parameters=True)
            dv = acceleration * dt
            dm = U[..., :1] * dv
            if trace:
                record.update(before_kick=U, acceleration=acceleration, force_scale=scale, force_mean=mean)
            if eos == "ideal":
                work = jnp.sum((U[..., momentum] + 0.5 * dm) * dv, axis=-1)
                U = U.at[..., energy].add(work)
            U = U.at[..., momentum].add(dm)
        if trace:
            record.update(primitives=W, saved=U_saved, conserved=U)
            stages.append(record)
        U, W = recover_primitives(U, eos, gamma, dfloor, pfloor)
    if trace:
        return U, tuple(stages)
    return U


def isothermal_timestep(W, dx, dy, cs, cfl, previous_dt, time, tlim, **kwargs):
    return hydro_timestep(W, (dx, dy), cfl, previous_dt, time, tlim,
                          eos="isothermal", cs=cs, **kwargs)


def isothermal_step(U, dt, dx, dy, cs, dfloor, integrator, **kwargs):
    return hydro_step(U, dt, (dx, dy), eos="isothermal", cs=cs, dfloor=dfloor,
                      integrator=integrator, **kwargs)
