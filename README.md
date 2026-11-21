# JathenAX

JathenAX evolves isothermal or ideal-gas hydrodynamics on a periodic 1D, 2D,
or 3D Cartesian grid on one JAX device. All active directions share one
finite-volume update. The default operators are PPM4 reconstruction, a Roe
Riemann solver, and RK3 integration. The ideal-gas solver conserves total
energy and supports linear thermal cooling and energy-normalized forcing.

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
python -m jathenax.run sod --nx 256 --output runs/sod.npz
python -m jathenax.run acoustic --nx 64 --ny 32 --nz 16 --output runs/acoustic.npz
python -m jathenax.run smooth --output runs/smooth.npz
python -m jathenax.run kh --output runs/kh.npz
python -m jathenax.run turbulence --nx 128 --ny 128 --tlim 1 --output runs/turbulence_smoke.npz
```

| Problem | Default grid and duration | Initial state |
| --- | --- | --- |
| `acoustic` | $64$, $t=1$ | Small right-moving sound wave in an ideal gas, one box crossing |
| `sod` | $256$, $t=0.1$ | Sod shock tube at the center and the reversed jump at the periodic seam, before the fans interact |
| `smooth` | $64\times48$, $t=0.2$ | Smooth periodic density and velocity waves |
| `kh` | $128^2$, $t=3$ | Two periodic Kelvin–Helmholtz shear layers marked by a tracer |
| `turbulence` | $256^2$, $t=45$ | Uniform density at rest, followed by seed-1618 forcing |

The default isothermal Kelvin–Helmholtz setup starts with uniform density and pressure
$\rho=p=1$. The inner band moves at $v_x=+0.5$ and carries tracer concentration
$s=1$. The outer layers move at $v_x=-0.5$ and carry $s=0$. The interfaces at
$y=\pm0.25$ are smoothed and seeded with a small transverse velocity
perturbation. The solver evolves $\rho s$ with the same mass flux as the gas.

Use `--integrator rk2`, `--integrator rk3`, or `--integrator rk4` to choose the
time integrator. Use `--viscosity` for the uniform kinematic coefficient and
`--smagorinsky-cs` for the dimensionless Smagorinsky coefficient. The
turbulence example defaults to uniform viscosity $1.5\times10^{-6}$ and drag
rate 0.3333335. Its forcing settings are in [inputs/turbulence.toml](inputs/turbulence.toml).

Use `--reconstruct` and `--rsolver` to select the face operators.

| Option | Choices |
| --- | --- |
| `--reconstruct` | `dc` (donor cell), `plm` (piecewise linear), `ppm4` (default), `ppmx` (extremum-preserving PPM), `wenoz` |
| `--rsolver` | `llf` (local Lax–Friedrichs), `hlle`, `hllc` (contact-resolving, ideal gas only), `roe` (default) |

```sh
python -m jathenax.run kh --reconstruct wenoz --rsolver hlle --output runs/kh_wenoz_hlle.npz
```

The same options are available as `hydro.reconstruct` and `hydro.rsolver` in a TOML deck.

The run prints the device, precision, final simulation time, and changes in
box-integrated conserved quantities. The first run includes JIT compilation.
Snapshots contain the active cell-center coordinates, `initial` and `conserved`,
the EOS and $\gamma$, the resolved input settings as JSON, and `time`, `dt`, and `cycle`. Grid shapes are
`(nx, nvar)`, `(ny, nx, nvar)`, or `(nz, ny, nx, nvar)`. The conserved
components are density, one momentum per active direction in $x,y,z$ order,
total energy for ideal gas, then any passive scalar densities. The existing
2D isothermal KH layout is $(\rho,m_x,m_y,\rho s)$.

From Python, `jathenax.hydro_step(U, dt, (dx, dy, dz), eos="ideal", gamma=5/3)`
advances a prescribed 3D step. Supply `(dx,)` or `(dx, dy)` on lower-dimensional
arrays. For an adaptive run, use `jathenax.run.evolve(U, spacing, cfl, tlim, nlim)`.

## Run from an input file

Copy [inputs/turbulence.toml](inputs/turbulence.toml) and edit it like an
AthenaK input deck. Set the uniform kinematic viscosity in `[hydro]` with
`viscosity` and the Smagorinsky coefficient with `smagorinsky_cs`. The deck
also sets the grid resolution, integrator, CFL number, drag, forcing, and
output path.

```sh
python -m jathenax.run -i inputs/turbulence.toml
python -m jathenax.run -i inputs/turbulence.toml --tlim 1 --viscosity 0 --output runs/inviscid.npz
```

The runner applies the selected problem's defaults, then the TOML settings,
then explicit command-line overrides. The periodic domain has unit length on
each active axis. In `[mesh]`, set `ny = 1` and `nz = 1` for 1D, set
`nz = 1` for 2D, and give all three sizes for 3D. Active axes need at least
six cells. Use `problem.axis = "x"`, `"y"`, or `"z"` to orient an acoustic
wave or shock tube. A 1D turbulence run defaults to compressive driving.
A deck that explicitly requests fully solenoidal forcing needs two or three dimensions.

Use `hydro.eos = "ideal"` for $P=(\gamma-1)e$, where $e$ is thermal energy
per volume, and set `hydro.gamma` and `hydro.pfloor` as needed.
`hydro_srcterms.tau_cool` applies $-e/\tau_{\rm cool}$ to total energy at
each RK stage. Zero disables cooling. To ramp the rate up from zero, set
`hydro_srcterms.cooling_ramp_time` to the time $T$ at which cooling reaches
full strength. Each stage multiplies the rate by $[x^2(3-2x)]^p$ with
$x=\min(1,\max(0,t/T))$. The default $T=0$ uses full strength from the start.
`hydro_srcterms.cooling_ramp_exponent` sets $p$. Its default of 1 gives half
strength at $T/2$, and 2 gives quarter strength there.
A forcing kick adds its full kinetic
work to total energy, preserving the cell's thermal energy.

The [Mach-0.5 pilot deck](inputs/mach05_3d.toml) starts a $32^3$ ideal-gas
box from rest with `problem.initial_sound_speed = 3`, unit driving power per
unit volume, solenoidal forcing, $\tau_{\rm cool}=14/3$, and zero explicit
viscosity and drag. The stationary run records the forcing work and thermal
loss at every step. [Measure stationarity and energy balance](validation/README.md#measure-the-native-stationary-box)

To initialize a turbulent velocity field, set `problem.initial_vrms` to its
3D mass-weighted rms speed and `problem.initial_velocity_seed` to the seed for
the initial draw. The field uses the driving band's mode weights and solenoidal
fraction. Its bulk momentum is removed and its kinetic energy is added to total
energy, preserving the requested initial sound speed.

```sh
JAX_PLATFORMS=cpu python -m validation.run_stationary_box inputs/mach05_3d.toml
```

The run stops at `time.tlim` or the positive cycle limit `time.nlim`, whichever comes
first. The output and checkpoint paths are relative to the working directory.

## Use a learned SGS stress

For 2D isothermal runs, supply a JAX predictor with `--sgs-model` and its checkpoint with
`--sgs-weights`. JathenAX calls the predictor on the full periodic density
and velocity fields at each Runge–Kutta stage and adds its stress to the
momentum face fluxes. See the [model interface and runnable example](examples/README.md).
The same settings can be written as `model` and `weights` in an `[sgs]` table.

The 3D ideal-gas pilot trains a first-derivative TBNN inside an $8^3$ Form A
solver. A live $32^3$ run supplies filtered targets and driving impulses. The
coarse solver predicts all eight comparison states before updating the weights.
The [online training instructions](validation/README.md#online-stress-training)
give the CPU and GPU commands, the checkpoint format, and the analytic comparison.

## Plot a run

```sh
python -m jathenax.plot_snapshot runs/kh.npz
python -m jathenax.spectra runs/kh.npz
```

The field plot shows profiles in 1D, maps in 2D and a central $z$ slice in 3D.
It includes ideal-gas pressure and the first tracer concentration when present.
The spectrum script
plots the final density-weighted kinetic energy and Fourier-curl enstrophy
in complete integer shells (kinetic energy alone in 1D) and writes a CSV with all shells, including the
zero mode.

## Code and verification

- [hydro.py](jathenax/hydro.py) handles time integration and timestep limits.
- [fluxes.py](jathenax/fluxes.py) holds reconstruction, Riemann, and viscous fluxes.
- [forcing.py](jathenax/forcing.py) builds the seeded OU sequence and normalizes kicks.
- [initial_conditions.py](jathenax/initial_conditions.py) defines the example problems.
- [test_periodic_hydro.py](tests/test_periodic_hydro.py) checks the acoustic wave, Sod solution, conservation, and energy sources.

See the [validation instructions](validation/README.md) for the AthenaK
stage, shear-decay, and forcing comparisons. The [design notes](DESIGN.md)
describe the numerical choices. This port retains the [AthenaK BSD notice](LICENSE-AthenaK).
