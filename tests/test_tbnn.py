import jax
import jax.numpy as jnp
import numpy as np

from jathenax.tbnn import realizability_scale, stress_inputs


def test_first_derivative_bases_on_periodic_shear():
    n, gamma, cs = 8, 5/3, 1.5
    phi = 2*np.pi*np.arange(n)/n
    s = np.zeros((n, n, n, 7))
    s[..., 0], s[..., 2], s[..., 6] = 2, 2*np.sin(phi), 0.35
    s[..., 5] = 2*cs**2/(gamma*(gamma-1))
    s[..., 4] = s[..., 5] + s[..., 2]**2/4 + s[..., 6]
    x = stress_inputs(jnp.asarray(s), gamma, 1e-8)
    g = np.cos(phi) * np.sin(2*np.pi/n)/cs
    strain, quad = np.zeros((n, 6)), np.zeros((n, 6))
    strain[:, 1] = g/2
    quad[:, [0, 3, 5]] = (g**2/3)[:, None] * [-1, 2, -1]
    np.testing.assert_allclose(x["basis"][0, 0], np.stack((strain, quad), axis=1), atol=2e-7)
    expected = np.column_stack((np.full(n, .35/(2*cs**2)), np.zeros(n), g**2/2, g**2/2))
    np.testing.assert_allclose(x["features"][0, 0], expected, atol=2e-7)
    vk = .175/(np.sqrt(.175+1e-8) + 1e-4)
    np.testing.assert_allclose(x["unit_eddy"][0, 0], -4*cs*vk*strain, atol=2e-7)


def test_realizability_at_zero_and_repeated_eigenvalues():
    d = jnp.asarray([[0., 0, 0, 0, 0, 0], [2., 0, 0, -1, 0, -1], [-2., 0, 0, 1, 0, 1],
                     [0., 1, 0, 0, 0, 0]], dtype=jnp.float32)
    k = jnp.array([0., .3, .3, .3], dtype=d.dtype)
    scale = realizability_scale(d, k)
    np.testing.assert_allclose(scale, [1., .2, .1, .2], atol=8e-5)
    gradient = jax.grad(lambda t: jnp.sum(t * realizability_scale(t, k)[:, None]))(d)
    assert np.isfinite(gradient).all()


def test_batched_realizability_gradient_on_rk_stages():
    rng = np.random.default_rng(1618)
    stress = rng.normal(size=(3, 32, 32, 32, 6)).astype(np.float32)
    stress[..., 5] = -stress[..., 0] - stress[..., 3]
    stress = jnp.asarray(stress)
    direction = jnp.asarray([0.3, -0.2, 0.4, 0.1, -0.5, -0.4], stress.dtype)

    def loss(t, tensor):
        shifted = tensor + t*direction
        scale = realizability_scale(shifted, jnp.ones_like(tensor[..., 0]))
        return (shifted[..., 0]*scale).mean()

    t = jnp.asarray(0.15, stress.dtype)
    reverse = jax.jit(jax.grad(loss))(t, stress)
    forward = jax.jit(jax.jacfwd(loss))(t, stress)
    np.testing.assert_allclose(reverse, forward, rtol=2e-5, atol=1e-6)
