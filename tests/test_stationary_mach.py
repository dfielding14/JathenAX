import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


def test_stationary_mach_removes_bulk_motion_and_weights_physical_time(tmp_path):
    deck = tmp_path / "pilot.athinput"
    deck.write_text("""<mesh>
nx1=32
nx2=32
nx3=32
x1min=0
x1max=2
x2min=0
x2max=1
x3min=0
x3max=1
<hydro>
gamma=1.6666666666666667
<problem>
tau_cool=10
<turb_driving>
dedt=0.05
rseed=1618
""")
    # Unit density, volume two, bulk speed 0.8, thermal density 0.4, and kinetic density 0.03+0.01t.
    t = np.array([0, 0.1, 0.4, 0.55, 0.9, 1.3, 2.1, 3.6, 4.4, 5.4, 6, 6])
    kinetic_lab = 2 * (0.03 + 0.01 * t) + 0.5 * 2 * 0.8**2
    rows = np.zeros((len(t), 10))
    rows[:, 0], rows[:, 2], rows[:, 3] = t, 2, 1.6
    rows[:, 6], rows[:, 7] = kinetic_lab + 0.8, kinetic_lab
    history = tmp_path / "pilot.hydro.hst"
    labels = "time dt mass 1-mom 2-mom 3-mom tot-E 1-KE 2-KE 3-KE".split()
    header = "Athena++ history data\n" + " ".join(f"[{i}]={label}" for i, label in enumerate(labels, 1))
    np.savetxt(history, rows, header=header)

    output = subprocess.check_output([sys.executable, "-m", "validation.stationary_mach",
                                      str(deck), str(history), "--interval", "0.5", "5.5",
                                      "--blocks", "2"], text=True)
    report = json.loads(output)
    np.testing.assert_allclose(report["mean_turbulent_kinetic_density"], 0.06)
    np.testing.assert_allclose(report["mean_thermal_density"], 0.4)
    np.testing.assert_allclose(report["mach"], np.sqrt(2 * 0.06 / ((10 / 9) * 0.4)))
    np.testing.assert_allclose(report["mean_cooling_over_power"], 0.8)
    np.testing.assert_allclose(report["storage_rate_over_power"], 0.2)
    np.testing.assert_allclose(report["history_budget_residual_over_power"], 0, atol=2e-15)
    np.testing.assert_allclose(report["tau_from_measured_kinetic"], 8.64)


@pytest.mark.parametrize(("options", "coarse_size", "steps", "vrms", "ramp_time"), [
    ((), 4, 4, 0., 0.), (("--replay-factor", "4", "--replay-steps", "5"), 2, 5, 0., 0.075),
    (("--spectrum-dt", "0.03"), 4, 4, 1.14, 0.),
])
def test_native_pilot_sound_speed_and_discrete_budget(tmp_path, options, coarse_size, steps, vrms, ramp_time):
    deck = tmp_path / "pilot.toml"
    base = Path(__file__).resolve().parents[1] / "inputs" / "mach05_3d.toml"
    text = base.read_text()
    for axis in "xyz":
        text = text.replace(f"n{axis} = 32", f"n{axis} = 8")
    cs = 2.28 if vrms else 3.0
    text = text.replace("initial_sound_speed = 3.0", f"initial_sound_speed = {cs}\ninitial_vrms = {vrms}")
    text = text.replace("tlim = 40.0", "tlim = 0.125")
    if ramp_time:
        text = text.replace("[hydro_srcterms]", f"[hydro_srcterms]\ncooling_ramp_time = {ramp_time}"
                                                "\ncooling_ramp_exponent = 2")
    deck.write_text(text[:text.index("[output]")] + '[output]\nfile = "final.npz"\n')
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "JAX_ENABLE_X64": "1",
           "PYTHONPATH": os.pathsep.join((str(base.parents[1]), os.environ.get("PYTHONPATH", "")))}
    subprocess.run([sys.executable, "-m", "validation.run_stationary_box", str(deck), "--replay-at", "0", *options],
                   cwd=tmp_path, env=env, check=True)
    window = subprocess.check_output([sys.executable, "-m", "validation.check_window",
                                      str(tmp_path / "window_000000")], env=env, text=True)
    checked = json.loads(window)
    assert checked["recorded_steps"] == steps and checked["coarse_cells_xyz"] == [coarse_size]*3
    subprocess.run([sys.executable, "-m", "jathenax.run", "-i", str(deck),
                    "--output", "ordinary.npz"], cwd=tmp_path, env=env, check=True)
    with np.load(tmp_path / "final.npz") as run, np.load(tmp_path / "ordinary.npz") as ordinary:
        initial = run["initial"]
        v = initial[..., 1:4]/initial[..., :1]
        np.testing.assert_allclose(v.mean(axis=(0,1,2)), 0, atol=2e-15)
        np.testing.assert_allclose(np.sqrt(np.mean(np.sum(v*v, axis=-1))), vrms, atol=2e-15)
        kinetic = 0.5*initial[..., 0]*np.sum(v*v,axis=-1)
        np.testing.assert_allclose(initial[..., 4]-kinetic, cs**2 / ((5/3)*(5/3 - 1)))
        np.testing.assert_array_equal(initial, ordinary["initial"])
        np.testing.assert_allclose(run["conserved"], ordinary["conserved"], atol=3e-13, rtol=0)
        assert run["cycle"] == ordinary["cycle"]
    if vrms:
        with np.load(tmp_path/"velocity_spectra.npz") as spectra:
            assert spectra["time"][0] == 0 and spectra["time"][-1] == 0.125
            assert np.all(np.diff(spectra["time"]) > 0)
            np.testing.assert_allclose(spectra["velocity_energy"][0].sum(), 0.5*vrms**2)
            np.testing.assert_allclose(spectra["velocity_energy"][0,4:], 0, atol=1e-25)
    output = subprocess.check_output(
        [sys.executable, "-m", "validation.stationary_mach", str(deck), "history.csv",
         "--interval", "0", "0.125", "--blocks", "2"], cwd=tmp_path, env=env, text=True,
    )
    report = json.loads(output)
    speed = np.sqrt(2 * report["mean_turbulent_kinetic_density"])  # Unit mean density.
    np.testing.assert_allclose(report["eddy_turnover_time"], 0.5 / speed)
    np.testing.assert_allclose(report["forcing_correlation_over_turnover"],
                               report["forcing_correlation_time"] * speed / 0.5)
    budget = report["measured_energy_budget"]
    if ramp_time:
        np.testing.assert_allclose(report["mean_cooling_over_power"], budget["cooling_over_drive"])
    np.testing.assert_allclose(budget["driving_work"], 0.125, rtol=1e-13)
    np.testing.assert_allclose(budget["energy_change"], budget["driving_work"] - budget["cooling_loss"],
                               atol=2e-13, rtol=0)


def test_filtered_favre_moments():
    from validation.replay_window import moments

    gamma = 5/3
    rho = np.linspace(0.5, 2, 8)
    velocity = np.arange(24, dtype=float).reshape(8, 3)**1.5 / 40
    thermal = np.linspace(3, 1, 8)
    mom = rho[:, None] * velocity
    total = thermal + (mom*velocity).sum(axis=1)/2
    u = np.column_stack((rho, mom, total))
    state, stress, heat, third = [np.asarray(a).ravel() for a in moments(u.reshape(2, 2, 2, 5), gamma)]
    dv = velocity - np.average(velocity, weights=rho, axis=0)
    k = 0.5*rho*np.sum(dv**2, axis=1)
    np.testing.assert_allclose(state, [*u.mean(axis=0), thermal.mean(), k.mean()], rtol=2e-6)
    np.testing.assert_allclose(stress, (dv.T @ (rho[:, None]*dv)/8)[np.triu_indices(3)], rtol=2e-6)
    np.testing.assert_allclose(heat, gamma*(thermal[:, None]*dv).mean(axis=0), rtol=2e-6)
    np.testing.assert_allclose(third, (k[:, None]*dv).mean(axis=0), rtol=2e-6)
