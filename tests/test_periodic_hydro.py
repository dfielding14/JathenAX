import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jathenax.fluxes import RECONSTRUCTIONS, RIEMANN_SOLVERS, hydro_flux, reconstruct_faces
from jathenax.forcing import normalize_force, periodic_basis, render_force
from jathenax.hydro import hydro_step, hydro_timestep, recover_primitives
from jathenax.initial_conditions import ACOUSTIC_AMPLITUDE, initial_state
from jathenax.run import evolve


ROUND = 48 * np.finfo(np.float64 if jax.config.x64_enabled else np.float32).eps


def centers(sizes):
    return tuple((np.arange(n) + 0.5) / n - 0.5 for n in sizes)


@pytest.mark.parametrize(("sizes", "modes_count"), [((11,), 17), ((11, 8), 17),
                                                      ((11, 8, 7), 17), ((11, 8, 7), 145)])
def test_driving_matches_direct_fourier_sum(sizes, modes_count):
    rng = np.random.default_rng(1620)
    ndim = len(sizes)
    coords = centers(sizes)
    modes = rng.integers(-3, 4, (modes_count, ndim))
    amplitudes = rng.normal(size=(len(modes), ndim, 2))
    basis = periodic_basis(modes, tuple(map(jnp.asarray, coords)), (1.,)*ndim)
    field = jax.jit(render_force)(jnp.asarray(amplitudes), basis)
    grids = np.meshgrid(*coords[::-1], indexing="ij")[::-1]
    phase = 2*np.pi * sum(grid[..., None]*modes[:, i] for i, grid in enumerate(grids))
    expected = np.cos(phase) @ amplitudes[..., 0] - np.sin(phase) @ amplitudes[..., 1]
    np.testing.assert_allclose(field, expected, rtol=ROUND, atol=np.sqrt(modes_count)*ROUND)


def wave_error(sizes, normal, tlim):
    coords = centers(sizes)
    initial = initial_state("acoustic", coords, "ideal", 5/3, 1.0, "xyz"[normal])
    U, time, _, _ = evolve(jnp.asarray(initial), tuple(1/n for n in sizes), 0.4, tlim, 1000)
    _, W = recover_primitives(U, "ideal", 5/3, 1e-10, 1e-10)
    axis = len(sizes) - 1 - normal
    fields = np.moveaxis(np.asarray(W), axis, 0).reshape(sizes[normal], -1, W.shape[-1])
    np.testing.assert_allclose(time, tlim)
    np.testing.assert_allclose(fields, np.broadcast_to(fields[:, :1], fields.shape), rtol=0, atol=ROUND)
    exact = ACOUSTIC_AMPLITUDE * np.sin(2*np.pi * (coords[normal] - tlim))
    rho, velocity, internal = fields[:, 0, 0], fields[:, 0, 1 + normal], fields[:, 0, -1]
    perturbations = (rho - 1, velocity, (5/3 - 1)*internal - 1/(5/3))
    return max(np.mean(np.abs(field - exact)) / ACOUSTIC_AMPLITUDE for field in perturbations)


@pytest.mark.parametrize(("sizes", "normal"), [
    ((32,), 0), ((32, 7), 0), ((6, 32), 1), ((32, 7, 6), 0),
    ((8, 32, 6), 1), ((7, 8, 32), 2),
])
def test_periodic_acoustic_wave_on_each_axis(sizes, normal):
    assert wave_error(sizes, normal, 0.25) < 0.01


def test_acoustic_refinement_over_a_period():
    coarse, fine = (wave_error((n,), 0, 1.0) for n in (32, 64))
    assert 0 < fine < 0.35 * coarse


@pytest.mark.parametrize("reconstruct", ("plm", "ppm4", "wenoz"))
@pytest.mark.parametrize("rsolver", ("roe", "hlle", "hllc"))
def test_periodic_sod_against_exact_solution(reconstruct, rsolver):
    # Standard gamma=1.4 Sod star state from the pressure-matching Riemann equation.
    gamma, p_star, u_star = 1.4, 0.303130178050647, 0.92745262004895
    x, = centers((256,))
    U0 = initial_state("sod", (x,), "ideal", gamma, 1.0)
    U, time, _, _ = evolve(jnp.asarray(U0), (1/len(x),), 0.4, 0.1, 1000, gamma=gamma,
                           reconstruct=reconstruct, rsolver=rsolver)
    _, W = recover_primitives(U, "ideal", gamma, 1e-10, 1e-10)
    # At t=0.1 the second Riemann problem at the periodic seam is outside this interval.
    interior = np.abs(x) < 0.25
    xi = x[interior] / float(time)
    a_left = np.sqrt(gamma)
    a_tail = a_left * p_star**((gamma - 1)/(2*gamma))
    ratio = (gamma - 1)/(gamma + 1)
    rho_star_r = 0.125 * (p_star/0.1 + ratio) / (ratio * p_star/0.1 + 1)
    shock = np.sqrt(((gamma + 1)*p_star + (gamma - 1)*0.1)/(2*0.125))
    regions = [xi < -a_left, xi < u_star - a_tail, xi < u_star, xi < shock]
    u_fan = 2.0/(gamma + 1) * (a_left + xi)
    a_fan = 1.0 - 0.5*(gamma - 1) * u_fan/a_left
    rho = np.select(regions, [1.0, a_fan**(2/(gamma - 1)), p_star**(1/gamma), rho_star_r], 0.125)
    vel = np.select(regions, [0.0, u_fan, u_star, u_star], 0.0)
    pres = np.select(regions, [1.0, a_fan**(2*gamma/(gamma - 1)), p_star, p_star], 0.1)
    actual = np.asarray(W)[interior] * [1, 1, gamma - 1]
    errors = np.mean(np.abs(actual - np.column_stack((rho, vel, pres))), axis=0)
    assert np.max(errors) < 0.012
    np.testing.assert_allclose(U.mean(axis=0), U0.mean(axis=0), rtol=0, atol=ROUND)


@pytest.mark.parametrize(("method", "min_order"), [("plm", 1.9), ("ppm4", 2.0), ("wenoz", 4.8)])
def test_reconstructed_periodic_faces_converge(method, min_order):
    errors = []
    for n in (16, 32):
        phase = 2*np.pi*np.arange(n)/n + 0.27
        averages = np.sinc(1/n)*np.sin(phase + np.pi/n)
        faces = RECONSTRUCTIONS[method](jnp.asarray(averages), 0)
        errors.append(max(np.mean(abs(np.asarray(q) - np.sin(phase))) for q in faces))
    assert errors[0]/errors[1] > 2**min_order


def test_wenoz_floors_accept_a_dynamic_negative_energy_index():
    w = jnp.broadcast_to(jnp.array([-0.1, -3.0, -0.2]), (8, 3))
    faces = jax.jit(lambda q, e: reconstruct_faces(q, 0, "wenoz", 0.5, energy_index=e, efloor=0.25))
    for side in faces(w, -1):
        np.testing.assert_array_equal(side, np.broadcast_to([0.5, -3.0, 0.25], w.shape))


@pytest.mark.parametrize("normal", [0, 1, 2])
def test_hllc_contact_shear_and_tracer(normal):
    gamma, pressure = 1.4, 1.0
    for vn in (0., -0.375, 0.5, 4.0):
        vl, vr = np.array([0.1, 0.3, -0.4]), np.array([-0.6, 0.4, 0.8])
        vl[normal] = vr[normal] = vn
        rl, rr = 1., 2.
        def side(rho, v, tracer):
            return jnp.array([rho, *v, pressure/(gamma-1), tracer]).reshape(1, 1, 1, -1)
        flux = hydro_flux(side(rl, vl, 0.2), side(rr, vr, 0.8), 2-normal,
                          eos="ideal", gamma=gamma, cs=1., rsolver="hllc")
        rho, v, tracer = (rl, vl, 0.2) if vn >= 0 else (rr, vr, 0.8)
        momentum = rho*vn*v
        momentum[normal] += pressure
        energy = (gamma*pressure/(gamma-1) + 0.5*rho*np.dot(v,v))*vn
        np.testing.assert_allclose(np.asarray(flux).ravel(), [rho*vn, *momentum, energy, rho*vn*tracer],
                                   rtol=ROUND, atol=ROUND)


def test_hllc_colliding_and_expanding_streams():
    gamma, speed, p = 1.4, 8.0, 0.4
    solve = jax.jit(lambda l, r: hydro_flux(l, r, 0, eos="ideal", gamma=gamma, cs=1., rsolver="hllc"))
    # Identical opposing streams have zero mass and energy flux by reflection symmetry.
    for sign in (1, -1):
        left = jnp.array([[1., sign*speed, p/(gamma-1)]])
        right = left.at[..., 1].multiply(-1)
        flux = solve(left, right)
        np.testing.assert_allclose(np.asarray(flux)[0, [0, 2]], 0, atol=ROUND)
        # At a stationary contact AthenaK transmits just the clipped PVRS contact pressure.
        sound = np.sqrt(gamma*p)
        p_mid = p + sign*speed*sound
        shock_factor = np.sqrt(1 + (gamma+1)/(2*gamma)*max(p_mid/p-1, 0))
        expected = max(p + sign*speed*sound*shock_factor, 0)
        np.testing.assert_allclose(flux[0, 1], expected, rtol=ROUND, atol=ROUND)
    x, = centers((160,))
    U0 = jnp.asarray(np.stack((np.ones_like(x), np.where(x < 0, -speed, speed),
                               np.full_like(x, p/(gamma-1) + 0.5*speed**2)), axis=-1))
    for tlim in (0.001, 0.005):
        U, time, _, _ = evolve(U0, (1/len(x),), 0.36, tlim, 1000, gamma=gamma, rsolver="hllc")
        _, W = recover_primitives(U, "ideal", gamma, 1e-10, 1e-10)
        assert np.isclose(time, tlim) and np.min(W[..., 0]) > 1e-8
        np.testing.assert_allclose(U.mean(axis=0)[:2], U0.mean(axis=0)[:2], atol=ROUND, rtol=ROUND)
        if tlim == 0.001:
            assert np.min(W[..., 2]) > 1e-8
            np.testing.assert_allclose(U.mean(axis=0)[2], U0.mean(axis=0)[2], rtol=ROUND)
        else:
            # The expanding fan reaches the pressure floor, which adds conserved energy.
            np.testing.assert_allclose((gamma-1)*np.min(W[..., 2]), 1e-10, rtol=ROUND)
            assert U.mean(axis=0)[2] > U0.mean(axis=0)[2]


def test_hllc_float32_derivative_at_stationary_collision():
    w = jnp.array([[1., 8., 1.]], dtype=jnp.float32)
    r = w.at[..., 1].multiply(-1)
    def loss(l):
        return hydro_flux(l, r, 0, eos="ideal", gamma=1.4, cs=1., rsolver="hllc").sum()
    assert np.isfinite(jax.jit(jax.grad(loss))(w)).all()


def test_isothermal_hllc_is_rejected():
    w = jnp.array([[1.7, -0.4]])
    with pytest.raises(ValueError, match="HLLC requires an ideal-gas EOS"):
        hydro_flux(w, w, 0, eos="isothermal", gamma=None, cs=1.3, rsolver="hllc")


def test_hlle_float32_gradient_when_roe_sound_speed_rounds_to_zero():
    w = jnp.array([[1., 1e6, 1.5]], dtype=jnp.float32)
    def mass_flux(l):
        return hydro_flux(l, w, 0, eos="ideal", gamma=5/3, cs=1., rsolver="hlle")[0, 0]
    # Both waves move right, so the mass flux equals rho_L * v_L.
    np.testing.assert_allclose(jax.jit(jax.grad(mass_flux))(w), [[1e6, 1., 0.]], atol=2e-6)


@pytest.mark.parametrize("rsolver", RIEMANN_SOLVERS)
@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_periodic_conservation_and_passive_scalar(ndim, rsolver):
    sizes = (10, 8, 6)[:ndim]
    U0 = initial_state("smooth", centers(sizes), "ideal", 1.4, 1)
    U0 = np.concatenate((U0, 0.3 * U0[..., :1]), axis=-1)
    U = hydro_step(jnp.asarray(U0), 0.005, tuple(1/n for n in sizes), gamma=1.4,
                   rsolver=rsolver, viscosity=0.02, smagorinsky_cs=0.1)
    axes = tuple(range(ndim))
    assert np.max(np.abs(U - U0)) > 1e-5
    np.testing.assert_allclose(U.mean(axis=axes), U0.mean(axis=axes), rtol=0, atol=ROUND)
    np.testing.assert_allclose(U[..., -1], 0.3*U[..., 0], rtol=0, atol=ROUND)


def test_ideal_energy_floor_and_multidimensional_cfl():
    U = jnp.asarray([[[[0.125, 0.5, 0.0, 1.0, -2.0, 0.25]]]])
    corrected, W = recover_primitives(U, "ideal", 1.5, 0.25, 0.5)
    np.testing.assert_array_equal(corrected[0, 0, 0], [0.25, 0.5, 0, 1.0, 3.5, 0.25])
    np.testing.assert_array_equal(W[0, 0, 0], [0.25, 2.0, 0, 4.0, 1.0, 1.0])
    dt = hydro_timestep(W, (0.5, 1.0, 0.1), 0.4, np.inf, 0.0, 1.0, gamma=1.5)
    np.testing.assert_allclose(dt, 0.4 * 0.1 / (4.0 + np.sqrt(3.0)))


def test_cooling_and_kick_close_energy_in_every_cell():
    sizes, dt, tau, dedt = (8, 7, 6), 0.02, 14.0, 0.03703705
    U0 = initial_state("smooth", centers(sizes), "ideal", 5/3, 1)
    force = jnp.asarray(np.random.default_rng(1618).normal(size=U0[..., 1:4].shape))
    final, stages = jax.jit(lambda u: hydro_step(
        u, dt, tuple(1/n for n in sizes), tau_cool=tau, raw_force=force, dedt=dedt, trace=True,
    ))(jnp.asarray(U0))
    before = stages[-1]["before_kick"]
    _, w_pre = recover_primitives(before, "ideal", 5/3, 1e-10, 1e-10)
    _, w_post = recover_primitives(final, "ideal", 5/3, 1e-10, 1e-10)
    np.testing.assert_allclose(w_pre[..., 4], w_post[..., 4], rtol=0, atol=ROUND)
    np.testing.assert_allclose((final[..., 4] - before[..., 4]).mean(), dt*dedt, atol=ROUND)
    weights = (1/6, 1/6, 2/3)
    loss = sum(w * s["primitives"][..., 4].mean() for w, s in zip(weights, stages)) * dt/tau
    np.testing.assert_allclose((final[..., 4] - U0[..., 4]).mean(), dt*dedt - loss, atol=ROUND)


@pytest.mark.parametrize("ndim", (1, 2, 3))
def test_isothermal_grid_preserves_uniform_tracer(ndim):
    sizes = (10, 8, 6)[:ndim]
    U0 = initial_state("smooth", centers(sizes), "isothermal", 5/3, 1)
    U0 = np.concatenate((U0, 0.2 * U0[..., :1]), axis=-1)
    U, _, _, _ = evolve(jnp.asarray(U0), tuple(1/n for n in sizes), 0.4, 0.04, 20, eos="isothermal")
    axes = tuple(range(ndim))
    np.testing.assert_allclose(U.mean(axis=axes), U0.mean(axis=axes), atol=ROUND)
    np.testing.assert_allclose(U[..., -1], 0.2 * U[..., 0], rtol=0, atol=ROUND)


def test_uniform_cooling_rk3():
    gamma, tau, dt = 5/3, 0.5, 0.04
    state = jnp.broadcast_to(jnp.array([1., 0.3, -0.2, 0.1, 1.07]), (6, 7, 8, 5))
    U = hydro_step(state, dt, (1/8, 1/7, 1/6), gamma=gamma, tau_cool=tau)
    h = dt/tau
    np.testing.assert_array_equal(U[..., :4], state[..., :4])
    np.testing.assert_allclose(U[..., 4], 0.07 + 1 - h + h*h/2 - h**3/6, rtol=ROUND)


@pytest.mark.parametrize(("exponent", "midpoint"), [(1, 0.5), (2, 0.25)])
def test_cooling_ramp_uses_rk_stage_times_and_saturates(exponent, midpoint):
    state = jnp.broadcast_to(jnp.array([1., 0., 1.]), (8, 3))
    dt, tau = 0.04, 0.5
    ramp = dict(cooling_ramp_time=dt, cooling_ramp_exponent=exponent)
    cooled, stages = hydro_step(state, dt, (1/8,), tau_cool=tau, trace=True, **ramp)
    np.testing.assert_allclose([s["cooling_fraction"] for s in stages], [0., 1., midpoint], atol=ROUND)
    h = dt/tau
    np.testing.assert_allclose(cooled[..., 2], 1/3 + (2/3)*(1 - h/4)*(1 - midpoint*h), rtol=ROUND)
    full = hydro_step(state, dt, (1/8,), tau_cool=tau)
    np.testing.assert_array_equal(hydro_step(state, dt, (1/8,), time=dt, tau_cool=tau, **ramp), full)


def test_transverse_viscous_shear_heats_the_gas():
    # A periodic shear v_y(x) has constant inviscid pressure and zero divergence.
    sizes, nu = (32, 7, 6), 0.025
    x, _, _ = centers(sizes)
    shear = 0.3 * np.sin(2*np.pi*x)
    U0 = np.zeros((*sizes[::-1], 5))
    U0[..., 0], U0[..., 2], U0[..., 4] = 1.0, shear, 1.0 + 0.5*shear**2
    U, t, _, _ = evolve(jnp.asarray(U0), tuple(1/n for n in sizes), 0.4, 0.05, 1000, viscosity=nu)
    _, W = recover_primitives(U, "ideal", 5/3, 1e-10, 1e-10)
    k2 = (2*sizes[0]*np.sin(np.pi/sizes[0]))**2
    np.testing.assert_allclose(U[0, 0, :, 2], shear*np.exp(-nu*k2*float(t)), atol=3e-4)
    kinetic = 0.5 * np.sum(np.asarray(U[..., 1:4])**2 / np.asarray(U[..., :1]), axis=-1).mean()
    heating = float(W[..., 4].mean()) - 1
    assert heating > 0
    np.testing.assert_allclose(heating, 0.5*np.mean(shear**2) - kinetic, atol=ROUND)


def test_delayed_3d_driving_has_zero_gradient_before_start():
    U = jnp.ones((6, 7, 8, 5))
    force = jnp.zeros_like(U[..., 1:4])
    grad = jax.grad(lambda u: jnp.sum(normalize_force(u, force, 0.01, 0.0)))(U)
    np.testing.assert_array_equal(grad, 0.0)
