"""Plot shell-integrated kinetic energy and enstrophy from a final snapshot."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--output", type=Path, help="Figure path, e.g. spectra.png")
    args = parser.parse_args()
    output = args.output or args.snapshot.with_name(args.snapshot.stem + "_spectra.png")
    if output.suffix == ".csv":
        parser.error("--output must be a figure path")

    with np.load(args.snapshot) as snapshot:
        rho, mx, my = np.moveaxis(snapshot["conserved"][..., :3].astype(np.float64), -1, 0)
        time = float(snapshot["time"])
    ny, nx = rho.shape
    kx, ky = np.fft.fftfreq(nx) * nx, np.fft.fftfreq(ny) * ny
    radius = np.hypot(kx[None, :], ky[:, None])
    shell = np.floor(radius + 0.5).astype(int)
    mode = np.arange(shell.max() + 1)

    vx, vy = mx / rho, my / rho
    root_rho = np.sqrt(rho)
    ax = np.fft.fft2(root_rho * vx) / rho.size
    ay = np.fft.fft2(root_rho * vy) / rho.size
    energy = np.bincount(shell.ravel(), (0.5 * (np.abs(ax)**2 + np.abs(ay)**2)).ravel())
    # Omit the axis Nyquist mode when taking a real Fourier derivative.
    if nx % 2 == 0:
        kx[nx // 2] = 0
    if ny % 2 == 0:
        ky[ny // 2] = 0
    omega = 2j * np.pi * (kx[None, :] * np.fft.fft2(vy) - ky[:, None] * np.fft.fft2(vx))
    enstrophy = np.bincount(shell.ravel(), (0.5 * np.abs(omega / rho.size)**2).ravel())
    np.testing.assert_allclose(energy.sum(), 0.5 * np.mean(rho * (vx**2 + vy**2)), rtol=1e-12)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output.with_suffix(".csv"), np.column_stack((mode, energy, enstrophy)),
               delimiter=",", header="mode,kinetic_energy,enstrophy", comments="",
               fmt=("%d", "%.16e", "%.16e"))
    complete = slice(1, min(nx, ny) // 2)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6), layout="constrained")
    for ax, values, label in zip(axes, (energy, enstrophy), ("Kinetic energy", "Enstrophy")):
        ax.loglog(mode[complete], values[complete])
        ax.set(xlabel="Box mode", ylabel=label)
        ax.grid(alpha=0.2)
    fig.suptitle(f"t = {time:g}")
    fig.savefig(output, dpi=160)
    print(output)
