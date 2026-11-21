"""Restrict a native periodic RK3 step to aligned cubic blocks of fine cells."""

import jax.numpy as jnp


def blocks(a, factor=2):
    z, y, x = a.shape[:3]
    if factor < 1 or any(n % factor for n in (z, y, x)):
        raise ValueError("the filter factor must be positive and divide each grid dimension")
    return a.reshape(z // factor, factor, y // factor, factor, x // factor, factor, *a.shape[3:])


def restrict(a, factor=2):
    return blocks(a, factor).mean(axis=(1, 3, 5))


def moments(u, gamma, factor=2):
    """Conserved and thermal state, Favre stress (xx,xy,xz,yy,yz,zz), q, and q3."""
    fine = blocks(u, factor)
    mean = fine.mean(axis=(1, 3, 5))
    rho, mom = fine[..., :1], fine[..., 1:4]
    velocity = mom / rho
    bulk = mean[..., 1:4] / mean[..., :1]
    fluct = velocity - bulk[:, None, :, None, :, None, :]
    thermal = fine[..., 4:5] - 0.5 * (mom * velocity).sum(axis=-1, keepdims=True)
    avg = (1, 3, 5)
    internal = thermal.mean(axis=avg)
    specific = internal / mean[..., :1]
    enthalpy_flux = gamma * ((thermal - rho * specific[:, None, :, None, :, None, :]) * fluct).mean(axis=avg)
    stress = jnp.stack([(rho[..., 0] * fluct[..., i] * fluct[..., j]).mean(axis=avg)
                        for i in range(3) for j in range(i, 3)], axis=-1)
    turbulent = 0.5 * rho * (fluct**2).sum(axis=-1, keepdims=True)
    state = jnp.concatenate((mean, internal, turbulent.mean(axis=avg)), axis=-1)
    return state, stress, enthalpy_flux, (turbulent * fluct).mean(axis=avg)


def step_record(initial, final, stages, dt, tau, gamma, factor=2):
    """Time-integrate fine fluxes and sources independently of the coarse state change."""
    weights = (1/6, 1/6, 2/3)
    states = (initial, stages[0]["conserved"], stages[1]["conserved"])
    stage_fields = [moments(u, gamma, factor) for u in states]
    state, stress, enthalpy, third = (jnp.stack(a) for a in zip(*stage_fields))

    faces = []
    for normal, name in enumerate("xyz"):
        flux = dt * sum(w * s[f"flux_{name}"] for w, s in zip(weights, stages))
        # Select the low boundary of each block and average its in-plane fine faces.
        fine_axis = 2 * (2 - normal) + 1
        plane = [slice(None)] * 7
        plane[fine_axis] = slice(0, 1)
        faces.append(blocks(flux, factor)[tuple(plane)].mean(axis=(1, 3, 5)))

    kick = stages[-1]
    pre = kick["before_kick"]
    dv = dt * kick["acceleration"]
    impulse = pre[..., :1] * dv
    work = ((pre[..., 1:4] + 0.5 * impulse) * dv).sum(axis=-1)
    loss = (dt / tau) * sum(w * s.get("cooling_fraction", 1.0) * restrict(s["primitives"][..., 4], factor)
                            for w, s in zip(weights, stages))
    return dict(filter_factor=factor, stage_state=state, final_state=moments(final, gamma, factor)[0],
                stage_stress=stress, stage_enthalpy_flux=enthalpy, stage_q3=third,
                face_flux_dt=jnp.stack(faces), pre_kick=restrict(pre, factor),
                impulse=restrict(impulse, factor), work=restrict(work, factor), cooling_loss=loss,
                force_scale=kick["force_scale"], force_mean=kick["force_mean"])
