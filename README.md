# JathenAX

JathenAX is a two-dimensional, doubly periodic, isothermal hydro solver in
JAX. It ports AthenaK's isothermal reconstruction and Riemann solvers,
with PPM4 and Roe as the defaults, and uses RK2, RK3, or RK4 time integration.
The grid runs on one JAX device. Uniform viscosity, Smagorinsky
viscosity, linear drag, seeded turbulence driving, and passive scalars are
available.

## Install

Use Python 3.12. Clone the project, then pick one environment for your
machine. Each installs the solver and the plotting tools.

```sh
git clone https://github.com/dfielding14/JathenAX.git
cd JathenAX
```

### Laptop CPU on macOS or Linux

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
```

### Apple silicon GPU

Use macOS 14.4 or newer and the Xcode Command Line Tools. The pinned
Metal environment runs in float32. [Apple Metal setup and supported types](https://developer.apple.com/metal/jax/)

```sh
python3.12 -m venv .venv-metal
source .venv-metal/bin/activate
python -m pip install -r requirements-metal.txt
export JAX_PLATFORMS=METAL JAX_ENABLE_X64=0
```

### NVIDIA H200 on Linux

Run on a GPU node with one H200 visible. Check the driver with `nvidia-smi`.
The CUDA 13 wheels require a Linux NVIDIA driver version of at least 580.
For a driver between 525 and 579, use `jax[cuda12]` in the install command.
JAX's pip extras supply the CUDA libraries. [JAX CUDA installation](https://docs.jax.dev/en/latest/installation.html#nvidia-gpu) · [NVIDIA GPU compute capabilities](https://developer.nvidia.com/cuda/gpus)

```sh
nvidia-smi
python3.12 -m venv .venv-cuda
source .venv-cuda/bin/activate
python -m pip install -r requirements.txt 'jax[cuda13]'
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=1
```

Verify the backend and precision in the activated environment.

```sh
python -c 'import jax; import jax.numpy as jnp; print(jax.devices(), jnp.ones(1).dtype)'
```

## Run an example

```sh
python -m jathenax.run smooth --output runs/smooth.npz
python -m jathenax.run kh --output runs/kh.npz
python -m jathenax.run turbulence --nx 128 --ny 128 --tlim 1 --output runs/turbulence_smoke.npz
```

| Problem | Default grid and duration | Initial state |
| --- | --- | --- |
| `smooth` | $64\times48$, $t=0.2$ | Smooth periodic density and velocity waves |
| `kh` | $128^2$, $t=3$ | Two periodic Kelvin–Helmholtz shear layers marked by a tracer |
| `turbulence` | $256^2$, $t=45$ | Uniform density at rest, followed by seed-1618 forcing |

The Kelvin–Helmholtz setup starts with uniform density and pressure
$\rho=p=1$. The inner band moves at $v_x=+0.5$ and carries tracer concentration
$s=1$. The outer layers move at $v_x=-0.5$ and carry $s=0$. The interfaces at
$y=\pm0.25$ are smoothed and seeded with a small transverse velocity
perturbation. The solver evolves $\rho s$ with the same mass flux as the gas.

Use `--integrator rk2`, `--integrator rk3`, or `--integrator rk4` to choose the
time integrator. Use `--viscosity` for the uniform kinematic coefficient and
`--smagorinsky-cs` for the dimensionless Smagorinsky coefficient. The
turbulence example defaults to uniform viscosity $1.5\times10^{-6}$ and drag
rate 0.3333335. Its forcing settings are in [run.py](jathenax/run.py).

Use `--reconstruct` and `--rsolver` to select the face operators.

| Option | Choices |
| --- | --- |
| `--reconstruct` | `dc` (donor cell), `plm` (piecewise linear), `ppm4` (default), `ppmx` (extremum-preserving PPM), `wenoz` |
| `--rsolver` | `llf` (local Lax–Friedrichs), `hlle`, `roe` (default) |

```sh
python -m jathenax.run kh --reconstruct wenoz --rsolver hlle --output runs/kh_wenoz_hlle.npz
```

The run prints the device, precision, final simulation time, and changes in
box-integrated conserved quantities. The first run includes JIT compilation.
Each output file contains cell centers `x` and `y`, initial and final states
`initial` and `conserved` in `(ny, nx, nvar)` order, and `time`, `dt`,
and `cycle`. The first three components are $(\rho,\rho v_x,\rho v_y)$.
The KH run appends the tracer density $\rho s$ as the fourth component.

## Plot a run

```sh
python -m jathenax.plot_snapshot runs/kh.npz
python -m jathenax.spectra runs/kh.npz
```

The field plot shows initial and final density, vorticity, and the first
tracer concentration where present, with shared color scales. The spectrum script
plots the final density-weighted kinetic energy and Fourier-curl enstrophy
in complete integer shells and writes a CSV with all shells, including the
zero mode.

## Code and verification

- [hydro.py](jathenax/hydro.py) handles time integration and timestep limits.
- [fluxes.py](jathenax/fluxes.py) holds reconstruction, Riemann, and viscous fluxes.
- [forcing.py](jathenax/forcing.py) builds the seeded OU sequence and normalizes kicks.
- [initial_conditions.py](jathenax/initial_conditions.py) defines the shear layers.

See the [validation instructions](validation/README.md) for the AthenaK
stage, shear-decay, and forcing comparisons. The [design notes](DESIGN.md)
describe the numerical choices. This port retains the [AthenaK BSD notice](LICENSE-AthenaK).
