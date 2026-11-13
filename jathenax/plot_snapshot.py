"""Plot the initial and final density, vorticity, and KH tracer."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--output", type=Path, help="Figure path, e.g. fields.png")
    args = parser.parse_args()
    output = args.output or args.snapshot.with_name(args.snapshot.stem + "_fields.png")

    with np.load(args.snapshot) as snapshot:
        x, y = snapshot["x"], snapshot["y"]
        initial, final = (snapshot[key].astype(np.float64) for key in ("initial", "conserved"))
        time = float(snapshot["time"])
    dx, dy = x[1] - x[0], y[1] - y[0]
    extent = (x[0] - dx / 2, x[-1] + dx / 2, y[0] - dy / 2, y[-1] + dy / 2)
    fields = []
    for U in (initial, final):
        rho, mx, my = np.moveaxis(U[..., :3], -1, 0)
        vx, vy = mx / rho, my / rho
        dvy_dx = (np.roll(vy, -1, axis=1) - np.roll(vy, 1, axis=1)) / (2 * dx)
        dvx_dy = (np.roll(vx, -1, axis=0) - np.roll(vx, 1, axis=0)) / (2 * dy)
        values = [rho, dvy_dx - dvx_dy]
        if U.shape[-1] > 3:
            values.insert(0, U[..., 3] / rho)
        fields.append(values)

    rho_min = min(field[-2].min() for field in fields)
    rho_max = max(field[-2].max() for field in fields)
    omega_max = max(np.abs(field[-1]).max() for field in fields)
    rows = [
        ("Density", "viridis", rho_min, rho_max),
        ("Vorticity", "RdBu_r", -omega_max, omega_max),
    ]
    if initial.shape[-1] > 3:
        rows.insert(0, ("Tracer concentration", "cividis", 0.0, 1.0))
    fig, axes = plt.subplots(len(rows), 2, figsize=(8, 3.7 * len(rows)),
                             sharex=True, sharey=True, layout="constrained")
    for row, (label, cmap, lo, hi) in enumerate(rows):
        for col, ax in enumerate(axes[row]):
            image = ax.imshow(fields[col][row], origin="lower", extent=extent, cmap=cmap,
                              vmin=lo, vmax=hi, interpolation="nearest")
        fig.colorbar(image, ax=axes[row], label=label, shrink=0.85)
    axes[0, 0].set_title("Initial")
    axes[0, 1].set_title(f"t = {time:g}")
    fig.supxlabel("x")
    fig.supylabel("y")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    print(output)
