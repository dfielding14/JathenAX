"""Compare native 3D spin-up with the four archived window_3d reference steps."""

import argparse
from configparser import ConfigParser
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jathenax.initial_conditions import initial_state
from jathenax.input import read_input
from jathenax.run import evolve, prepare_forcing


def compare(directory):
    """Run from rest with native adaptive timesteps and restrict only the final fields."""
    deck = ConfigParser(interpolation=None, inline_comment_prefixes=("#",))
    deck.read_string(Path(__file__).with_name("window_3d.athinput").read_text()
                     .replace("<", "[").replace(">", "]"))
    defaults = read_input(None, "turbulence")["turb_driving"]
    driving = {k: type(v)(deck["turb_driving"].get(k, v)) for k, v in defaults.items()}
    sizes = tuple(deck.getint("mesh", k) for k in ("nx1", "nx2", "nx3"))
    spacing = tuple(1/n for n in sizes)
    coords = tuple((np.arange(n) + 0.5) / n - 0.5 for n in sizes)
    gamma = deck.getfloat("hydro", "gamma")
    initial = jnp.asarray(initial_state("turbulence", coords, "ideal", gamma, 1.0))
    # A stop time past cycle 12 leaves every native CFL step, including the last, unclipped.
    tlim = 0.2
    forcing = prepare_forcing(driving, coords, tlim, initial.dtype)
    peak = np.zeros(5)
    for cycle in range(8, 12):
        with (directory / f"window.{cycle}.bin").open("rb") as file:
            header = file.readline().split()
            nx, ny, nz, r = map(int, header[:4])
            dx, dy, dz, time, dt, dedt, tau, ref_gamma = map(float, header[4:])
            target = np.fromfile(file, dtype=np.float64).reshape(34, nz, ny, nx)[7:12]
        np.testing.assert_equal((nx*r, ny*r, nz*r), sizes)
        np.testing.assert_allclose((dx/r, dy/r, dz/r), spacing)
        np.testing.assert_equal((dedt, tau, ref_gamma), (driving["dedt"], 2.0, gamma))
        U, native_time, native_dt, steps = evolve(
            initial, spacing, 0.4, tlim, cycle + 1, gamma=gamma, forcing=forcing,
            dt_update=driving["dt_update"], dedt=dedt, tau_cool=tau,
        )
        avg = np.asarray(U).reshape(nz, r, ny, r, nx, r, 5).mean(axis=(1, 3, 5))
        reference = np.moveaxis(target, 0, -1)
        np.testing.assert_equal(int(steps), cycle + 1)
        np.testing.assert_allclose((native_time, native_dt), (time + dt, dt), rtol=0, atol=2e-15)
        np.testing.assert_allclose(avg, reference, rtol=0, atol=5e-14)
        peak = np.maximum(peak, np.max(np.abs(avg - reference), axis=(0, 1, 2)))
    return peak


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Reference output from validation/window_3d.athinput")
    jax.config.update("jax_enable_x64", True)
    errors = compare(parser.parse_args().directory)
    print("Peak absolute differences across cycles 8–11")
    for field, err in zip(("density", "x momentum", "y momentum", "z momentum", "total energy"), errors):
        print(f"  {field:14s} {err:.3e}")
