"""Plot initial and final profiles, planar fields, or a central z slice."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def snapshot_fields(U, coords, eos, gamma):
    ndim = len(coords)
    if ndim == 3:
        U = U[len(coords[2]) // 2]
    rho = U[..., 0]
    velocity = U[..., 1:1 + ndim] / rho[..., None]
    fields = {"Density": rho}
    if ndim == 1:
        fields["Velocity"] = velocity[..., 0]
    else:
        dx, dy = (v[1] - v[0] for v in coords[:2])
        vx, vy = velocity[..., 0], velocity[..., 1]
        dvy_dx = (np.roll(vy, -1, axis=1) - np.roll(vy, 1, axis=1)) / (2 * dx)
        dvx_dy = (np.roll(vx, -1, axis=0) - np.roll(vx, 1, axis=0)) / (2 * dy)
        fields["Vorticity (z)"] = dvy_dx - dvx_dy
    ngas = 1 + ndim
    if eos == "ideal":
        fields["Pressure"] = (gamma - 1) * (U[..., ngas] - 0.5 * rho * np.sum(velocity**2, axis=-1))
        ngas += 1
    if U.shape[-1] > ngas:
        fields = {"Tracer concentration": U[..., ngas] / rho, **fields}
    return fields


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--output", type=Path, help="Figure path, e.g. fields.png")
    args = parser.parse_args()
    output = args.output or args.snapshot.with_name(args.snapshot.stem + "_fields.png")

    with np.load(args.snapshot) as snapshot:
        coords = [snapshot[k] for k in "xyz" if k in snapshot]
        eos = str(snapshot["eos"]) if "eos" in snapshot else "isothermal"
        gamma = float(snapshot["gamma"]) if "gamma" in snapshot else 5/3
        labels = ("Initial", f"t = {float(snapshot['time']):g}")
        initial, final = [snapshot_fields(snapshot[k].astype(np.float64), coords, eos, gamma)
                          for k in ("initial", "conserved")]

    if len(coords) == 1:
        fig, axes = plt.subplots(len(initial), 1, figsize=(7, 2 * len(initial)),
                                 sharex=True, layout="constrained")
        for ax, label in zip(axes, initial):
            ax.plot(coords[0], initial[label], "--", label=labels[0])
            ax.plot(coords[0], final[label], label=labels[1])
            ax.set_ylabel(label)
        axes[0].legend()
    else:
        x, y = coords[:2]
        dx, dy = x[1] - x[0], y[1] - y[0]
        extent = (x[0] - dx/2, x[-1] + dx/2, y[0] - dy/2, y[-1] + dy/2)
        fig, axes = plt.subplots(len(initial), 2, figsize=(8, 3.7 * len(initial)),
                                 sharex=True, sharey=True, layout="constrained")
        for row, label in enumerate(initial):
            pair = (initial[label], final[label])
            lo, hi = min(v.min() for v in pair), max(v.max() for v in pair)
            cmap = "viridis"
            if label == "Vorticity (z)":
                hi = max(abs(lo), abs(hi))
                lo, cmap = -hi, "RdBu_r"
            elif label == "Tracer concentration":
                lo, hi, cmap = 0.0, 1.0, "cividis"
            for ax, values in zip(axes[row], pair):
                image = ax.imshow(values, origin="lower", extent=extent, cmap=cmap,
                                  vmin=lo, vmax=hi, interpolation="nearest")
            fig.colorbar(image, ax=axes[row], label=label, shrink=0.85)
        for ax, title in zip(axes[0], labels):
            ax.set_title(title)
        if len(coords) == 3:
            fig.suptitle(f"z = {coords[2][len(coords[2]) // 2]:g}")
        fig.supylabel("y")
    fig.supxlabel("x")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    print(output)
