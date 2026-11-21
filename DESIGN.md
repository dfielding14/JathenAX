# JathenAX design

## Target

JathenAX is a finite-volume solver for Newtonian hydro on a uniform, periodic
Cartesian grid in one, two, or three dimensions. The whole grid lives on one JAX device. The reference is
AthenaK's [Tiegan SGS commit c50d3d5e9](https://github.com/dfielding14/athenak/commit/c50d3d5e962cf4c576de88c7b5dea33f4d2674dc).
Use a fresh AthenaK run on that commit for comparison. It applies a single
forcing kick per timestep, whereas the earlier seed-1618 pair used stage kicks.

Keep the implementation readable from top to bottom. `hydro.py` owns primitive
recovery, RK, and timestep selection. `fluxes.py` owns face operators,
`forcing.py` owns the turbulence driver, and `run.py` owns initialization,
the time loop, and output. Keep the example settings together in `run.py`,
with AthenaK's parameter names where the semantics match.

## State and face fluxes

Spatial array axes run from the last active coordinate to the first, with
components last. A 3D ideal-gas state has shape `(nz, ny, nx, 5)` and
components $(\rho,m_x,m_y,m_z,E)$. Lower dimensions carry one momentum per
active axis. Isothermal states omit $E$. Both EOSs can append passive
scalar densities $\rho s$. The isothermal law is $p=c_{\rm iso}^2\rho$.
For ideal gas, recover $e=E-|\boldsymbol m|^2/(2\rho)$ and $p=(\gamma-1)e$.
Density and pressure floors preserve the cell momenta. Raising the thermal
energy to its floor also raises conserved $E$.

Reconstruct density, velocity, ideal-gas internal energy density, and tracer
concentrations along each active axis from the same incoming RK stage. A
face carries the index of the cell immediately above it. The low face comes
from that cell and its periodic neighbor. Each passive tracer flux equals
the Riemann mass flux times its upwind concentration.

| Operator | Implementation |
| --- | --- |
| Reconstruction | `dc`, `plm`, `ppm4`, `ppmx`, and `wenoz`. PPMX and WENO-Z apply the face density and ideal-gas pressure floors. |
| Riemann solver | `llf`, `hlle`, and `roe` for both equations of state. Ideal gas also supports `hllc`. Roe falls back to LLF for negative intermediate density. |
| Viscosity | Uniform kinematic viscosity and optional Smagorinsky viscosity. Face stresses use the physical $2/3$ trace subtraction in every dimension. The ideal-gas energy flux includes stress work at the mean face velocity. |

HLLC uses AthenaK's primitive-variable pressure estimate for the outer waves and
clips the contact pressure at zero. In both HLLC and HLLE, a left speed at or
above zero becomes $-10^{-20}$ and a right speed at or below zero becomes $10^{-20}$.
HLLE estimates the ideal-gas Roe sound speed from the averaged enthalpy.
WENO-Z uses AthenaK's $\epsilon=10^{-42}$. In FP32 it rescales the smoothness
indicators and epsilon together to keep the weights and their gradients finite.

## Time integration

| Input | Method |
| --- | --- |
| `rk2` | Two-stage SSPRK(2,2) |
| `rk3` | Three-stage SSPRK(3,3) |
| `rk4` | AthenaK's four-stage, two-register low-storage RK4 |

At the beginning of each timestep, set the saved register $V$ to the conserved
state $U$. Before each subsequent RK4 stage, accumulate $V\leftarrow
V+\delta_s U$ using the current stage's incoming $U$. Then apply

$$
U \leftarrow \gamma_{0,s}U+\gamma_{1,s}V-\beta_s\Delta t\,D_hF.
$$

Use [AthenaK's stage coefficients](https://github.com/dfielding14/athenak/blob/c50d3d5e962cf4c576de88c7b5dea33f4d2674dc/src/driver/driver.cpp#L99-L160)
and the deck's `cfl_number`. The reference driver records nominal limits of 1.0
for RK2 and RK3 and 1.3925 for RK4. The mesh timestep selector uses the configured
number directly. The stage uses the sum of the hyperbolic and enabled diffusion face
fluxes. Apply Rayleigh drag after the flux update with the same stage weight and
the incoming stage primitives. Apply the turbulence kick after the final stage's
drag and before final primitive recovery. Recover primitives and apply floors
after every stage.

Compute the next timestep from the post-kick state. Take the minimum of
$\Delta x_i/(|v_i|+c)$ over cells and active directions, with $c^2=\gamma p/\rho$
for ideal gas. Apply the diffusion, drag, and cooling bounds before the CFL
factor, then the factor-of-two growth cap and end-time clipping.

For $d$ active directions, let $h=\min_i \Delta x_i$ and
$\Delta=(\prod_i \Delta x_i)^{1/d}$. The uniform viscous bound is
$h^2/(2d\nu)$. With Smagorinsky enabled, use
$3h^2/[8d(\nu+2\max_{\rm faces}\nu_{\rm SGS})]$ with
$\nu_{\rm SGS}=C_s^2\Delta^2\sqrt{2S_{ij}S_{ij}}$. Recompute the bound from the
current state at each timestep and the flux from the incoming state at each RK stage.
Ideal-gas linear cooling uses $-e/\tau_{\rm cool}$ evaluated on those same stage primitives.

## Turbulence driving

1. Build the Cartesian mode list in AthenaK order. Port `RanSt` and
   `RanGaussianSt`, with the same real and imaginary draw order even for modes
   whose weight is zero. Apply the same spectrum, solenoidal projection, and OU
   recurrence. Build the finite run's modal coefficient table once at startup
   and transfer it to the GPU. This port supports continuous Cartesian
   parabolic driving.
2. Select the modes due at the cycle-start time. Sum the same signed modes at
   AthenaK's cell centers using a real matrix multiply at the simulation precision.
   The summation order can differ at roundoff. Cache the raw field between OU updates.
3. For each full-timestep kick, subtract the current mass-weighted mean
   acceleration. Normalize with the post-stage conserved state. In `edot`
   mode, match the quadratic solve so the box receives
   $\texttt{dedt}\,V_{\rm box}\Delta t$ of kinetic energy. Apply the
   momentum kick. The future `accel_rms` option sets the volume-weighted RMS
   acceleration. Ideal gas adds the full finite kinetic-energy change to $E$.

The first driven target is the seed-1618 deck at $256^2$, with isothermal
PPM4/Roe/RK3, uniform viscosity $1.5\times10^{-6}$, drag rate $0.3333335$,
solenoidal Cartesian modes $n=3$–$5$ peaked at $n=4$, $\dot e=0.03703705$,
`tcorr=0.75`, `dt_update=0.0075`, and `tlim=45`. Use one AthenaK MeshBlock
and one rank for the parity run.

## Learned stress in the same flux update

Keep the model as a JAX function of its weights and the stage's density and
in-plane velocity fields. The weights and flow arrays are placed on the same
GPU once. With a whole-grid input, the model can use any receptive field its
architecture and device memory support. Match the padding used in training
at the periodic seam.

The model returns cell-centered $(\tau_{xx},\tau_{xy},\tau_{yy})$ in code units,
with shape `(ny, nx, 3)`. The solver evaluates it once per RK stage and
averages adjacent predictions onto periodic faces. It adds the positive
Favre stress to the momentum flux. The RK update takes the negative divergence.
The [model interface](examples/README.md) shows how to load a checkpoint and
run the predictor on a snapshot or inside the solver.

```python
tau = predict_stress(weights, primitives[..., :3])
tau_x, tau_y = stress_face_fluxes(tau)
flux_x = flux_x.at[..., 1:3].add(tau_x)
flux_y = flux_y.at[..., 1:3].add(tau_y)
```

The 2D hook uses the existing isothermal equations and timestep limits.

The periodic 3D pilot advances density, momentum, total energy $\overline E$,
and SGS kinetic-energy density $K$. At $\gamma=5/3$, the effective Euler
pressure is $2(U+K)/3$, where $U$ is thermal energy density. PPM4/Roe supplies
the face mass flux that advects $K/R$. Each face uses the same deviatoric stress
and velocity for momentum transport, total-energy work, and SGS production.
Cooling removes $U$ and SGS decay converts $K$ into $U$ at fixed $\overline E$.
The saved and live training paths use the same RK3 step and apply the same
filtered impulses at their recorded times. [3D LES step](jathenax/les.py) ·
[Differentiable trajectory](jathenax/online.py)

## Build and verification order

1. **Current scaffold.** Verify all three RK schedules on a linear test equation
   and verify stress sign, component order, periodic wrapping, and conservation
   with a manufactured stress field inside JIT.
2. **Hydro parity.** AthenaK supplies reference fluxes for LLF, HLLE and Roe
   with either EOS and for ideal-gas HLLC. The comparisons cover RK2, RK3, and RK4.
   The full-precision AthenaK fixture compares face states, fluxes, and RK
   registers for a smooth flow and a velocity jump at a prescribed timestep.
   See the [reproduction commands](validation/README.md#verify-against-athenak).
3. **Viscosity.** Uniform and Smagorinsky momentum fluxes and timestep limits
   pass the [periodic shear checks](validation/README.md#verify-periodic-shear-decay).
   Ideal-gas tests also check total energy and viscous heating. Thermal conduction remains a future extension.
4. **Forcing parity.** The [seed-1618 fixture](validation/README.md#verify-seed-1618-forcing)
   compares mode indices, RNG draws, modal amplitudes, accelerations, and the
   pre- and post-kick states over 12 cycles with drag. It replays AthenaK's
   timesteps, then checks adaptive timesteps and energy and momentum injection.
5. **GPU run and solver menu.** Use the device-side timestep loop from the
   unforced example. Compare the $256^2$ pair's energy and enstrophy spectra
   over a common late-time window. The native ideal-gas step is checked against
   acoustic waves, the Sod solution, and a short 3D AthenaK reference window.
   Test the student's predictor on a saved snapshot before applying it in a
   short coupled run.

Compare float64 dumps for parity. AthenaK's standard binary snapshots store
float32 field data. Record the first bitwise mismatch, ULP distance, and a
scaled absolute error. Exact equality is a diagnostic on a pinned setup.
XLA and Kokkos can use different reductions and floating-point operation order,
so the GPU acceptance criterion is agreement to roundoff on early steps and
agreement of late-time statistics. Use a float64-capable GPU for the precision
comparison and reserve Apple Metal for float32 smoke checks.
