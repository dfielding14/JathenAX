"""Plot shell-integrated kinetic energy and enstrophy from a final snapshot."""

import argparse
from pathlib import Path

import numpy as np


def mode_shells(shape):
    frequencies = [np.fft.fftfreq(n)*n for n in shape]
    waves = np.meshgrid(*frequencies, indexing="ij", sparse=True)
    return np.floor(np.sqrt(sum(k*k for k in waves)) + 0.5).astype(int)


def shell_energy(components, shell):
    """Half the squared vector amplitude, summed in integer Fourier shells."""
    power = 0.5 * sum(np.abs(np.fft.fftn(v) / shell.size)**2 for v in components)
    return np.bincount(shell.ravel(), power.ravel())


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--output", type=Path, help="Figure path, e.g. spectra.png")
    args = parser.parse_args()
    output = args.output or args.snapshot.with_name(args.snapshot.stem + "_spectra.png")
    if output.suffix == ".csv":
        parser.error("--output must be a figure path")

    with np.load(args.snapshot) as snapshot:
        U = snapshot["conserved"].astype(np.float64)
        time = float(snapshot["time"])
    rho = U[..., 0]
    ndim = rho.ndim
    velocity = [U[..., i + 1] / rho for i in range(ndim)]
    frequencies = [np.fft.fftfreq(n) * n for n in rho.shape]
    shell = mode_shells(rho.shape)
    mode = np.arange(shell.max() + 1)
    root_rho = np.sqrt(rho)
    energy = shell_energy((root_rho*v for v in velocity), shell)
    # The real Fourier derivative vanishes at each even axis's Nyquist mode.
    for k in frequencies:
        if len(k) % 2 == 0:
            k[len(k) // 2] = 0
    k = np.meshgrid(*frequencies, indexing="ij", sparse=True)[::-1]
    vhat = [np.fft.fftn(v) / rho.size for v in velocity]
    enstrophy_density = np.zeros_like(rho)
    for i in range(ndim):
        for j in range(i + 1, ndim):
            curl = 2j * np.pi * (k[i] * vhat[j] - k[j] * vhat[i])
            enstrophy_density += 0.5 * np.abs(curl)**2
    enstrophy = np.bincount(shell.ravel(), enstrophy_density.ravel())
    np.testing.assert_allclose(energy.sum(), 0.5 * np.mean(rho * sum(v**2 for v in velocity)),
                               rtol=1e-12)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output.with_suffix(".csv"), np.column_stack((mode, energy, enstrophy)),
               delimiter=",", header="mode,kinetic_energy,enstrophy", comments="",
               fmt=("%d", "%.16e", "%.16e"))
    complete = slice(1, min(rho.shape) // 2)
    spectra = [(energy, "Kinetic energy")]
    if ndim > 1:
        spectra.append((enstrophy, "Enstrophy"))
    fig, axes = plt.subplots(1, len(spectra), figsize=(4.25 * len(spectra), 3.6), layout="constrained")
    for ax, (values, label) in zip(np.atleast_1d(axes), spectra):
        ax.loglog(mode[complete], values[complete])
        ax.set(xlabel="Box mode", ylabel=label)
        ax.grid(alpha=0.2)
    fig.suptitle(f"t = {time:g}")
    fig.savefig(output, dpi=160)
    print(output)
