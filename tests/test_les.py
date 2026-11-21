import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jathenax.hydro import hydro_step
from jathenax.input import read_input
from jathenax.les import GAMMA, Transport, apply_impulse, filtered_state, les_rhs, les_step
from jathenax.online import rollout
from validation.evaluate_online_tbnn import long_summary, means, record_reference
from validation.online_reference import live_blocks, training_batch


def state(n, sgs_energy, speed):
    x = (jnp.arange(n) + 0.5) / n
    vx = jnp.broadcast_to(speed*jnp.sin(2*jnp.pi*x), (n, n, n))
    density = jnp.ones_like(vx)
    return jnp.stack((density, vx, 0*vx, 0*vx, 3 + sgs_energy + 0.5*vx**2,
                      sgs_energy*density), axis=-1)


def test_reference_targets_after_completed_cooling_ramp(tmp_path):
    cfg = read_input(Path(__file__).resolve().parents[1] / "inputs" / "mach05_3d.toml")
    cfg["mesh"].update(nx=8, ny=8, nz=8)
    cfg["hydro_srcterms"]["cooling_ramp_time"] = 0.02
    centers = (np.arange(8) + 0.5)/8 - 0.5
    snapshot, output = tmp_path / "fine.npz", tmp_path / "reference"
    np.savez(snapshot, conserved=state(8, 0., 0.2)[..., :5], x=centers, y=centers, z=centers,
             time=0.025, dt=0.005, settings=json.dumps(cfg))
    record_reference(snapshot, output, blocks=1)
    manifest = json.loads((output / "reference.json").read_text())
    assert (manifest["fine_n"], manifest["coarse_n"], manifest["short_blocks"]) == (8, 2, [0])
    with np.load(output / "block_000.npz") as block:
        np.testing.assert_allclose(block["target_time"], np.arange(3, 11)/100, rtol=0, atol=1e-14)
        np.testing.assert_allclose(block["drive_work"].sum(), 0.075, rtol=2e-6)
        # Old stationary checkpoints omit the ramp setting. Precompiling for timing
        # must give the same fully active cooling and the same fine driving kicks.
        del cfg["hydro_srcterms"]["cooling_ramp_time"]
        compile_times = {}
        timed, clock = next(live_blocks(snapshot, cfg, None, 1, 0.01, compile_times=compile_times))
        np.testing.assert_array_equal(timed["final"][clock[2]], block["target"])
        np.testing.assert_array_equal(timed["impulse"], block["impulse"])
        assert compile_times["fine_compile_seconds"] > 0
    cfg["hydro_srcterms"]["cooling_ramp_time"] = 0.05
    with np.testing.assert_raises_regex(ValueError, "cooling at full strength"):
        next(live_blocks(snapshot, cfg, None, 1, 0.01))


def test_gas_limit_and_matched_kick():
    q = state(8, 0., 0.3)
    dt = 0.002
    model = dict(k0=1e-8, c_nu=0.1)
    step = jax.jit(lambda q: les_step(q, dt, 1/8, model, Transport(0.2, 0.4, 0.6), 4.,
                                      stress="analytic"))
    result, _, minima = step(q)
    gas = hydro_step(q[..., :5], dt, (1/8,)*3, gamma=GAMMA, tau_cool=4.)
    np.testing.assert_allclose(result[..., :5], gas, atol=2e-6)
    np.testing.assert_array_equal(result[..., 5], 0.)
    assert minima[2] == 0.
    q = state(8, 0.2, 0.4)
    impulse = jnp.full_like(q[..., 1:4], 0.03)
    after, work = apply_impulse(q, impulse, 0.007*jnp.ones_like(q[..., 5]))
    np.testing.assert_allclose(filtered_state(q)[..., 5], filtered_state(after)[..., 5], atol=1e-6)
    np.testing.assert_allclose(after[..., 4] - q[..., 4], work, atol=3e-7)


def test_subgrid_pressure_work_and_acoustic_speed():
    # At uniform rho,K this identity is material conservation of K/rho^(5/3).
    q = state(16, 0.35, 0.1)
    model, transport = dict(k0=1e-8), Transport(0., 0., 0.)
    fields = filtered_state(q)
    zero = jnp.zeros_like(q)
    div = les_rhs(fields, zero, 1/16, model, transport)
    np.testing.assert_allclose(div[..., 5], GAMMA*0.35*div[..., 0], atol=2e-5)
    # A linear right-going isentropic mode has dm/dt = c_eff * drho/dt.
    sound = np.sqrt(GAMMA*(GAMMA-1)*(3+0.35))
    wave = 1e-3*jnp.sin(2*jnp.pi*(jnp.arange(16)+0.5)/16)
    rho = 1 + jnp.broadcast_to(wave, q.shape[:3])
    v = sound*(rho-1)
    q = jnp.stack((rho, rho*v, v*0, v*0, 3.35*rho**GAMMA + 0.5*rho*v**2,
                   0.35*rho**GAMMA), axis=-1)
    div = les_rhs(filtered_state(q), zero, 1/16, model, transport)
    np.testing.assert_allclose(div[..., 1], sound*div[..., 0], atol=1.4e-4)
    np.testing.assert_allclose(div[..., :5].mean(axis=(0, 1, 2)), 0., atol=2e-7)


def test_rollout_parameter_gradient_and_initialization():
    # A nonuniform flow exercises the transposed PPM limiters across multiple RK steps.
    rng = np.random.default_rng(1618)
    q = np.array(state(4, 0.3, 0.2))
    q[..., 1:4] += 0.2*rng.normal(size=q[..., 1:4].shape)
    q = jnp.asarray(q)
    initial = filtered_state(q)
    stages = jnp.broadcast_to(initial, (4, 3, *initial.shape))
    data = dict(stage=stages, dt=jnp.full(4, 0.01, q.dtype),
                impulse=jnp.zeros((4, 4, 4, 4, 3), q.dtype), sgs_work=jnp.zeros((4, 4, 4, 4), q.dtype))
    probe = jnp.asarray(rng.normal(size=q.shape[:3]), q.dtype)

    def loss(coefficient, inputs):
        model = dict(gamma=GAMMA, c_nu=0.02, k0=1e-8, feature_mean=jnp.zeros(4, q.dtype),
                     feature_scale=jnp.ones(4, q.dtype),
                     params=dict(w1=jnp.zeros((4, 1), q.dtype), b1=jnp.zeros(1, q.dtype),
                                 w2=jnp.zeros((1, 2), q.dtype), b2=jnp.stack((coefficient, coefficient/3))))
        history, _, _, _ = rollout(model, inputs, Transport(0.03, 0.03, 0.4), 14/3)
        return (history[-1, ..., 1]*probe).sum()

    value = jax.jit(loss)
    theta = jnp.asarray(0.15, q.dtype)
    grad = jax.jit(jax.grad(loss))(theta, data)
    h = 0.002 if q.dtype == np.float32 else 1e-5
    centered = (value(theta+h, data)-value(theta-h, data))/(2*h)
    np.testing.assert_allclose(grad, centered, rtol=0.007, atol=1e-5)
    changed_targets = dict(data, stage=stages.at[1:].multiply(2))
    np.testing.assert_array_equal(value(theta, data), value(theta, changed_targets))

    # The fourth compiled slot can be empty when a live segment needs only three kicks.
    short = {key: field[:3] for key, field in data.items()}
    short["duration"] = short["dt"]
    short["impulse"] = short["impulse"].at[-1, ..., 0].set(0.03)
    short["sgs_work"] = short["sgs_work"].at[-1].set(0.007)
    padded = training_batch(short, 4)
    for key in ("dt", "duration", "impulse", "sgs_work"):
        np.testing.assert_array_equal(padded[key][3], 0.)
    np.testing.assert_array_equal(value(theta, short), value(theta, padded))


def test_continuous_rollout_keeps_state_and_accounts_for_sources():
    q = state(4, 0.25, 0.2)
    initial = filtered_state(q)
    kicks = dict(dt=jnp.full(4, 0.006, q.dtype), impulse=jnp.full((4, 4, 4, 4, 3), 0.001, q.dtype),
                 sgs_work=jnp.full((4, 4, 4, 4), 0.002, q.dtype))

    def advance(fields, data):
        return rollout(dict(k0=1e-8, c_nu=0.05), data, Transport(0.02, 0.03, 0.4), 14/3,
                       stress="analytic", initial=fields, return_sources=True)

    run = jax.jit(advance)
    full, _, _, _, sources = run(initial, kicks)
    first = run(initial, {k: v[:2] for k, v in kicks.items()})[0]
    last = run(first[-1], {k: v[2:] for k, v in kicks.items()})[0]
    np.testing.assert_allclose(last, full[2:], rtol=2e-6, atol=2e-7)
    change = (np.asarray(full[-1, ..., 4]) - np.asarray(initial[..., 4])).mean()
    np.testing.assert_allclose(change, np.asarray(sources[:, 0]-sources[:, 1]).sum(), atol=5e-7)
    history = means(np.concatenate((np.asarray(initial)[None], np.asarray(full))))
    times = np.r_[0., np.cumsum(np.asarray(kicks["dt"], dtype=float))]
    report = long_summary(times, history, *np.asarray(sources).sum(axis=0, dtype=float))
    assert abs(report["energy_residual_per_time"] * times[-1]) < 5e-7
    boost = np.asarray(initial).copy()
    boost[..., 1] += 2*boost[..., 0]
    np.testing.assert_allclose(means(boost)[2], history[0, 2], atol=1e-7)
