from math import factorial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jathenax.forcing import normalize_force
from jathenax.fluxes import (
    RIEMANN_SOLVERS, flux_divergence, isothermal_flux, stress_face_fluxes, wenoz_face_states,
)
from jathenax.hydro import (
    RK_STAGES,
    isothermal_step,
    isothermal_timestep,
    recover_isothermal,
    rk_stage,
)
from jathenax.initial_conditions import kelvin_helmholtz
from jathenax.run import evolve_isothermal


@pytest.mark.parametrize(("integrator", "order"), [("rk2", 2), ("rk3", 3), ("rk4", 4)])
def test_rk_stage_schedule(integrator, order):
    # The linear test equation dU/dt = U has a known stability polynomial.
    @jax.jit
    def step(U, dt):
        U_saved = U
        for coefficients in RK_STAGES[integrator]:
            U, U_saved = rk_stage(U, U_saved, -U, dt, coefficients)
        return U

    dt = 0.125
    result = step(jnp.asarray(1.0), dt)
    expected = sum(dt**n / factorial(n) for n in range(order + 1))
    np.testing.assert_allclose(result, expected, rtol=32 * np.finfo(result.dtype).eps)


def test_rk3_zero_flux_preserves_float32_density():
    rho = jnp.asarray(np.linspace(0.75, 1.25, 1024, dtype=np.float32))
    stage = jax.jit(
        lambda rho: rk_stage(rho, rho, jnp.zeros_like(rho), 0.01, RK_STAGES["rk3"][-1])[0]
    )
    np.testing.assert_array_equal(stage(rho), rho)


def test_density_floor_preserves_momentum():
    U, W = recover_isothermal(jnp.asarray([[[0.125, 0.5, -0.25]]]), dfloor=0.25)
    np.testing.assert_array_equal(U, [[[0.25, 0.5, -0.25]]])
    np.testing.assert_array_equal(W, [[[0.25, 2.0, -1.0]]])


def test_isothermal_timestep_limits():
    W = jnp.asarray([[[1.0, 1.0, 2.0], [1.0, 3.0, 4.0]]])
    # dx / (max|vx| + cs) is the tighter directional bound.
    assert isothermal_timestep(W, 1.0, 2.0, 1.0, 0.5, np.inf, 0.0, 1.0) == 0.125
    assert isothermal_timestep(W, 1.0, 2.0, 1.0, 0.5, 0.03125, 0.0, 1.0) == 0.0625
    assert isothermal_timestep(W, 1.0, 2.0, 1.0, 0.5, np.inf, 0.9375, 1.0) == 0.0625


def test_viscous_timestep_limits():
    W = jnp.asarray([[[1.0, 0.0, 0.0]] * 2, [[1.0, 1.0, 0.0]] * 2])
    dx, dy, cfl = 0.25, 0.5, 0.5
    nu, Cs = 0.5, 2.0
    timestep_args = (dx, dy, 1.0, cfl, np.inf, 0.0, 1.0)
    uniform_dt = isothermal_timestep(W, *timestep_args, viscosity=nu)
    assert uniform_dt == cfl * 0.25 * dx**2 / nu

    # The face gradient of vx across y faces has magnitude 1/dy.
    max_nu_sgs = Cs**2 * dx * dy / dy
    combined_dt = isothermal_timestep(W, *timestep_args, viscosity=nu, smagorinsky_cs=Cs)
    np.testing.assert_allclose(combined_dt, cfl * (3/16) * dx**2 / (nu + 2*max_nu_sgs))

    W_at_rest = W.at[..., 1].set(0.0)
    assert isothermal_timestep(W_at_rest, *timestep_args, smagorinsky_cs=Cs) == cfl * dx


@pytest.mark.parametrize(("integrator", "order"), [("rk2", 2), ("rk3", 3), ("rk4", 4)])
def test_uniform_linear_drag(integrator, order):
    initial = jnp.broadcast_to(jnp.asarray([2.0, 0.6, -0.4]), (6, 8, 3))
    dt, rate, cfl = 0.125, 4.0, 0.4
    _, W = recover_isothermal(initial, dfloor=1e-10)
    bound = isothermal_timestep(W, 1.0, 1.0, 1.0, cfl, np.inf, 0.0, 1.0, drag_rate=rate)
    np.testing.assert_allclose(bound, cfl / rate)
    final = isothermal_step(initial, dt, 1.0, 1.0, 1.0, 1e-10, integrator, drag_rate=rate)
    factor = sum((-rate * dt)**n / factorial(n) for n in range(order + 1))
    tolerance = 32 * np.finfo(final.dtype).eps
    np.testing.assert_allclose(final[..., 0], initial[..., 0], rtol=tolerance)
    np.testing.assert_allclose(final[..., 1:], initial[..., 1:] * factor, rtol=tolerance)


def test_kick_normalization():
    rho = jnp.asarray([[[1.0], [2.0]]])
    raw_force = jnp.asarray([[[2.0, 1.0], [-1.0, 1.0]]])
    dt, dedt = 0.125, 0.04
    for sign in (-1.0, 1.0):
        momentum = sign * 0.125 * rho * raw_force
        U = jnp.concatenate((rho, momentum), axis=-1)
        force = jax.jit(normalize_force)(U, raw_force, dt, dedt)
        delta = rho * force * dt
        energy = jnp.mean(0.5 * jnp.sum(delta * (2.0 * momentum + delta) / rho, axis=-1))
        tolerance = 64 * np.finfo(force.dtype).eps
        np.testing.assert_allclose(delta.sum(axis=(0, 1)), 0.0, atol=tolerance)
        np.testing.assert_allclose(energy, dt * dedt, rtol=tolerance)
    np.testing.assert_array_equal(normalize_force(U, raw_force, dt, 0.0), 0.0)


@pytest.mark.parametrize("integrator", RK_STAGES)
def test_periodic_step_conserves_mass_and_momentum(integrator):
    nx, ny = 8, 6
    x, y = np.meshgrid(2.0 * np.pi * np.arange(nx) / nx, 2.0 * np.pi * np.arange(ny) / ny)
    rho = 1.0 + 0.2 * np.cos(x + y)
    U = jnp.asarray(np.stack((rho, 0.2 * rho * np.sin(y), 0.3 * rho * np.cos(x)), axis=-1))
    step = jax.jit(lambda U: isothermal_step(U, 0.01, 1/nx, 1/ny, 1.0, 1e-10, integrator))
    final = np.asarray(step(U))
    initial = np.asarray(U)
    assert np.max(np.abs(final - initial)) > 1.0e-4
    np.testing.assert_allclose(
        final.mean(axis=(0, 1)), initial.mean(axis=(0, 1)),
        rtol=0, atol=16 * np.finfo(final.dtype).eps,
    )


@pytest.mark.parametrize("rsolver", RIEMANN_SOLVERS)
def test_passive_scalars_share_mass_flux_without_changing_hydro(rsolver):
    x = (np.arange(12) + 0.5) / 12 - 0.5
    U = kelvin_helmholtz(x, x)
    np.testing.assert_array_equal(U[..., 0], 1.0)
    assert U[..., 3].min() == 0.0 and U[..., 3].max() == 1.0
    # Give one tracer a uniform concentration even as the density evolves.
    U = jnp.asarray(np.concatenate((U, 0.25 * U[..., :1]), axis=-1))
    force = jnp.broadcast_to(jnp.sin(2 * jnp.pi * jnp.asarray(x))[None, :, None], (12, 12, 2))
    step = jax.jit(lambda U: isothermal_step(
        U, 0.01, 1/12, 1/12, 1.0, 1e-10, "rk3", viscosity=0.001,
        smagorinsky_cs=0.2, drag_rate=0.3, raw_force=force, dedt=0.04,
        rsolver=rsolver,
    ))
    final = np.asarray(step(U))
    tolerance = 32 * np.finfo(final.dtype).eps
    np.testing.assert_allclose(final[..., :3], step(U[..., :3]), rtol=0, atol=tolerance)
    np.testing.assert_allclose(final[..., 4], 0.25 * final[..., 0], rtol=0, atol=tolerance)
    np.testing.assert_allclose(final[..., 3:].mean(axis=(0, 1), dtype=np.float64),
                               [0.5, 0.25], rtol=0, atol=tolerance)
    assert np.max(np.abs(final[..., 3] - U[..., 3])) > 1e-4


def test_wenoz_float32_flat_stencils_at_jump():
    W = jnp.asarray([[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]], dtype=jnp.float32)
    left, right = jax.jit(lambda W: wenoz_face_states(W, 1))(W)
    np.testing.assert_allclose(left, jnp.roll(W, 1, axis=1), atol=2e-7)
    np.testing.assert_allclose(right, W, atol=2e-7)


def test_learned_stress_couples_conservatively_under_jit():
    nx, ny = 8, 5
    dx, dy = 1.0 / nx, 1.0 / ny
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dy
    k = 2.0 * np.pi
    vx, vy = np.meshgrid(np.sin(k * x), np.sin(k * y))
    primitives = jnp.asarray(np.stack((np.ones_like(vx), vx, vy), axis=-1))
    weights = jnp.asarray([0.5, 2.0, -1.0, 1.5])
    dt = 0.125

    # Pass weights as device arrays, as we would for the student's predictor.
    @jax.jit
    def momentum_increment(weights, primitives):
        vx, vy = primitives[..., 1], primitives[..., 2]
        tau = jnp.stack(
            (weights[0] * vx, weights[1] * vx + weights[2] * vy, weights[3] * vy),
            axis=-1,
        )
        flux_x, flux_y = stress_face_fluxes(tau)
        return -dt * flux_divergence(flux_x, flux_y, dx, dy)

    result = momentum_increment(weights, primitives)

    # Face averaging and differencing give the centered difference of each sine.
    dvx_dx = np.cos(k * x)[None, :] * np.sin(k * dx) / dx
    dvy_dy = np.cos(k * y)[:, None] * np.sin(k * dy) / dy
    div_x = 0.5 * dvx_dx - dvy_dy
    div_y = 2.0 * dvx_dx + 1.5 * dvy_dy
    expected = -dt * np.stack((div_x, div_y), axis=-1)

    tolerance = 32 * np.finfo(result.dtype).eps
    np.testing.assert_allclose(result, expected, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(result.sum(axis=(0, 1)), 0.0, atol=tolerance * nx * ny)


@pytest.mark.parametrize("integrator", RK_STAGES)
def test_sgs_predictor_uses_each_stage_and_conserves_momentum(integrator):
    x = (np.arange(8) + 0.5) / 8 - 0.5
    U = jnp.asarray(kelvin_helmholtz(x, x))
    weights = {"scale": jnp.asarray([0.0125, 0.025, -0.0375])}

    def predict_stress(weights, W):
        assert W.shape == (8, 8, 3)
        return weights["scale"] * W

    @jax.jit
    def step(U, weights):
        return isothermal_step(
            U, 0.01, 1/8, 1/8, 1.0, 1e-10, integrator, trace=True,
            predict_stress=predict_stress, sgs_weights=weights,
        )

    final, stages = jax.device_get(step(U, weights))
    baseline, _ = step(U, {"scale": jnp.zeros(3, dtype=U.dtype)})
    assert np.max(np.abs(final[..., 1:3] - baseline[..., 1:3])) > 1e-4
    tolerance = 16 * np.finfo(final.dtype).eps
    for stage in stages:
        tau = np.asarray(weights["scale"]) * stage["primitives"][..., :3]
        for axis, direction, components in ((1, "x", slice(0, 2)), (0, "y", slice(1, 3))):
            gas = isothermal_flux(stage[f"left_{direction}"], stage[f"right_{direction}"], 1.0, axis)
            stress = tau[..., components]
            expected = 0.5 * (stress + np.roll(stress, 1, axis=axis))
            added = stage[f"flux_{direction}"] - np.asarray(gas)
            np.testing.assert_allclose(added[..., 1:3], expected, rtol=0, atol=tolerance)
            np.testing.assert_allclose(added[..., [0, 3]], 0.0, rtol=0, atol=tolerance)
    np.testing.assert_allclose(final.mean(axis=(0, 1)), np.asarray(U).mean(axis=(0, 1)),
                               rtol=0, atol=tolerance)


def test_zero_sgs_predictor_matches_baseline_evolution():
    x = (np.arange(8) + 0.5) / 8 - 0.5
    U = jnp.asarray(kelvin_helmholtz(x, x))
    args = (U, 1/8, 1/8, 1.0, 1e-10, "rk3", 0.4, 0.1, 10)

    def zero_stress(weights, W):
        return jnp.zeros_like(W)

    baseline = evolve_isothermal(*args)
    coupled = evolve_isothermal(*args, predict_stress=zero_stress)
    for a, b in zip(baseline, coupled, strict=True):
        np.testing.assert_array_equal(a, b)


def test_sgs_predictor_requires_the_full_grid():
    U = jnp.ones((6, 8, 3))
    with pytest.raises(ValueError, match="predict_stress must return"):
        isothermal_step(U, 0.01, 1/8, 1/6, 1.0, 1e-10, "rk3",
                        predict_stress=lambda weights, W: W[:1])
