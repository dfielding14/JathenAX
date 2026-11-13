# JathenAX design

## Target

JathenAX is a finite-volume solver for 2D Newtonian hydro on a uniform, doubly
periodic Cartesian grid. The whole grid lives on one GPU. The reference is
AthenaK's [Tiegan SGS commit c50d3d5e9](https://github.com/dfielding14/athenak/commit/c50d3d5e962cf4c576de88c7b5dea33f4d2674dc).
Use a fresh AthenaK run on that commit for comparison. It applies a single
forcing kick per timestep, whereas the earlier seed-1618 pair used stage kicks.

Keep the implementation readable from top to bottom. `hydro.py` owns primitive
recovery, RK, and timestep selection. `fluxes.py` owns face operators,
`forcing.py` owns the turbulence driver, and `run.py` owns initialization,
the time loop, and output. Keep the example settings together in `run.py`,
with AthenaK's parameter names where the semantics match.

## State and face fluxes

The state shape is `(ny, nx, nvar)`, with components $(\rho,m_x,m_y)$ for
isothermal hydro and $(\rho,m_x,m_y,E)$ for ideal gas. The ideal-gas
primitives are $(\rho,v_x,v_y,e)$, where $e$ is internal energy per unit
volume. Match AthenaK's primitive recovery and its floor application order.
The isothermal law is $p=c_{\rm iso}^2\rho$.

The isothermal state can append passive scalar densities $\rho s$. Reconstruct
the concentrations $s$ with the selected method and multiply the upwind
face value by the Riemann solver's mass flux.

Reconstruct the primitives along both directions from the same RK stage.
Store one flux per periodic face. Face `[j, i]` is the low-coordinate face
of cell `[j, i]` in that flux's direction. Use wrapped shifts for neighbors
and swap normal and transverse components when solving along $y$. Form the
divergence by differencing the high and low faces in each direction.

| Operator | Planned choices |
| --- | --- |
| Reconstruction | Isothermal hydro supports `dc`, `plm`, `ppm4`, `ppmx`, and `wenoz`. PPMX and WENO-Z apply the face density floor. |
| Riemann solver | Isothermal hydro supports `llf`, `hlle`, and `roe`. Roe retains its LLF fallback for negative intermediate density. HLLC belongs with the ideal-gas solver. |
| Viscosity | Uniform kinematic viscosity plus optional `smagorinsky_cs`. The isothermal port uses face-averaged density, face velocity gradients, and $2/3$ trace subtraction. Ideal-gas viscous work belongs with the ideal-gas solver. |
| Conduction | Constant conductivity for ideal gas. Add $-\kappa\nabla(p/\rho)$ to the energy flux at faces and use AthenaK's parabolic timestep bound. |

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

Compute the next timestep from the post-kick state. Match the directional
minimum of $\Delta x/(|v_x|+c_s)$ and $\Delta y/(|v_y|+c_s)$ over the grid.
Include the viscous, conductive, and drag bounds, the factor-of-two growth cap,
and clipping at the final time. The forcing update times and snapshot times
leave the hydro timestep unchanged.

For viscosity, let $h=\min(\Delta x,\Delta y)$. Before applying the CFL
factor, the uniform bound is $h^2/(4\nu)$. With Smagorinsky enabled,
the bound is $(3/16)h^2/(\nu+2\max_{\rm faces}\nu_{\rm SGS})$. Compute
$\nu_{\rm SGS}=C_s^2\Delta x\,\Delta y\,\sqrt{2S_{ij}S_{ij}}$ on the
same faces used by the momentum flux. Recompute the bound from the current
state at each timestep and the flux from the incoming state at each RK stage.

## Turbulence driving

1. Build the Cartesian mode list in AthenaK order. Port `RanSt` and
   `RanGaussianSt`, with the same real and imaginary draw order even for modes
   whose weight is zero. Apply the same spectrum, solenoidal projection, and OU
   recurrence. Build the finite run's modal coefficient table once at startup
   and transfer it to the GPU. This port supports continuous Cartesian
   parabolic driving.
2. Select the modes due at the cycle-start time. Use AthenaK's cell centers
   and ordered mode summation to render the raw acceleration. Cache the raw
   field between OU updates.
3. For each full-timestep kick, subtract the current mass-weighted mean
   acceleration. Normalize with the post-stage conserved state. In `edot`
   mode, match the quadratic solve so the box receives
   $\texttt{dedt}\,V_{\rm box}\Delta t$ of kinetic energy. Apply the
   momentum kick. The future `accel_rms` option sets the volume-weighted RMS
   acceleration. Ideal gas will also add the kinetic-energy change to $E$.

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
with shape `(ny, nx, 3)`. Evaluate it once per RK stage. The scaffold already
averages adjacent predictions onto periodic faces. For the positive Favre
covariance convention used by Tiegan, add those results to the momentum flux
with a plus sign. The RK update takes the negative divergence.

```python
tau = predict_stress(weights, primitives[..., :3])
tau_x, tau_y = stress_face_fluxes(tau)
flux_x = flux_x.at[..., 1:3].add(tau_x)
flux_y = flux_y.at[..., 1:3].add(tau_y)
```

Start learned-stress tests with the isothermal equations. An ideal-gas LES
also needs a decision about the energy closure.

## Build and verification order

1. **Current scaffold.** Verify all three RK schedules on a linear test equation
   and verify stress sign, component order, periodic wrapping, and conservation
   with a manufactured stress field inside JIT.
2. **Hydro parity.** All five isothermal reconstructions and three Riemann
   solvers run with RK2, RK3, and RK4.
   The full-precision AthenaK fixture compares face states, fluxes, and RK
   registers for a smooth flow and a velocity jump at a prescribed timestep.
   See the [reproduction commands](validation/README.md#verify-against-athenak).
3. **Viscosity.** Uniform and Smagorinsky momentum fluxes and timestep limits
   pass the [periodic shear checks](validation/README.md#verify-periodic-shear-decay).
   Verify conduction and viscous work when the ideal-gas solver is available.
4. **Forcing parity.** The [seed-1618 fixture](validation/README.md#verify-seed-1618-forcing)
   compares mode indices, RNG draws, modal amplitudes, accelerations, and the
   pre- and post-kick states over 12 cycles with drag. It replays AthenaK's
   timesteps, then checks adaptive timesteps and energy and momentum injection.
5. **GPU run and solver menu.** Use the device-side timestep loop from the
   unforced example. Compare the $256^2$ pair's energy and enstrophy spectra
   over a common late-time window. Verify the ideal-gas choices with AthenaK
   when that solver is available.
   Test the student's predictor on a saved snapshot before applying it in a
   short coupled run.

Compare float64 dumps for parity. AthenaK's standard binary snapshots store
float32 field data. Record the first bitwise mismatch, ULP distance, and a
scaled absolute error. Exact equality is a diagnostic on a pinned setup.
XLA and Kokkos can use different reductions and floating-point operation order,
so the GPU acceptance criterion is agreement to roundoff on early steps and
agreement of late-time statistics. Use a float64-capable GPU for the precision
comparison and reserve Apple Metal for float32 smoke checks.
