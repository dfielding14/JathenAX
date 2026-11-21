# JathenAX validation

Run from the repository root after installing JathenAX. Install pytest for
the unit tests.

```sh
python -m pip install pytest
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python -m pytest -q
```

## Verify native periodic hydro

```sh
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python -m pytest -q tests/test_periodic_hydro.py
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python -m jathenax.run -i inputs/mach05_3d.toml --tlim 0.05 --output runs/mach05_smoke.npz
```

The wave tests advance the density, velocity, and pressure perturbations along each
active axis, including on rectangular 3D grids. The Sod test compares all three
fields to the exact solution between the center and the periodic images at $t=0.1$.
The source tests check isolated RK3 cooling, viscous heating, zero net impulse,
and finite kick work. The short example samples the startup of the native unit-power box.
The Sod comparison covers all nine PLM, PPM4, and WENO-Z pairings with Roe, HLLE, and HLLC.
Ideal-gas HLLC contact and shear tests check every face orientation and the passive
tracer flux. Opposing streams check the contact pressure in compression and expansion,
conservation before the pressure floor activates, and floor heating in a deep rarefaction.

After building the stage fixture below, pass `--ideal-3d` to
`validation.compare_athenak` to compare 3D ideal-gas primitives, reconstructed
states, fluxes and conserved RK registers. It runs one step on smooth and
velocity-jump data for RK2, RK3 and RK4. All Riemann solvers use the same
opposing velocities of $\pm2.5$ in the jump test. HLLC also runs the $\pm0.5$ case.
At opposing supersonic faces the
energy-flux check accounts for cancellation using each side's enthalpy density
times its maximum signal speed. The 3D pressure and vertical-velocity phases
are offset from exact grid symmetries. At tied smooth extrema, a one-ulp change
in the incoming primitive can select a different PPMX limiter branch.

The four archived $\tau_{\rm cool}=2$ steps from the reference-window fixture
also test the independently evolving native driver. Start JathenAX from rest
and compare its adaptive timestep and its $32^3$ solution restricted to the
same $8^3$ cells at each recorded endpoint.

```sh
JAX_PLATFORMS=cpu python -m validation.compare_ideal_window runs/reference_window_3d
```

## Verify against AthenaK

Use the AthenaK commit pinned in [DESIGN.md](../DESIGN.md), with its Kokkos
submodule initialized. From the JathenAX directory, build the reference
problem generator and run the comparison in a separate output directory.

```sh
ATHENAK=/path/to/athenak
CHECK_DIR=$(mktemp -d)
PGEN=$(python -c 'import os, sys; print(os.path.relpath("validation/athenak_stages", sys.argv[1] + "/src/pgen"))' "$ATHENAK")
cmake -S "$ATHENAK" -B "$CHECK_DIR/build" -DPROBLEM="$PGEN" -DCMAKE_BUILD_TYPE=Release
cmake --build "$CHECK_DIR/build" -j 8
JAX_PLATFORMS=cpu python -m validation.compare_athenak \
    "$CHECK_DIR/build/src/athena" "$CHECK_DIR/comparison"
```

The fixture runs one fixed step on a periodic $24\times16$ grid, using a
smooth flow and a velocity jump for each integrator. AthenaK writes incoming
primitives, face states evaluated with the selected reconstruction, actual Riemann fluxes,
the saved RK register, and updated conserved variables with 17 significant
digits. The comparison saves the JAX arrays and reports absolute and bitwise
differences in `stage_errors.csv`.

Add `--viscosity 0.001 --smagorinsky-cs 0.2` and use a fresh comparison
directory to check the viscous face fluxes on the same flows.
To check another pairing, add `--reconstruct wenoz --rsolver hlle` and use a
fresh comparison directory.

## Verify periodic shear decay

```sh
JAX_PLATFORMS=cpu python -m validation.periodic_shear /path/to/new/shear-output
```

The float64 RK3 check runs both viscosities on a periodic $256^2$ grid.
It checks the uniform-viscosity decay against the discrete shear solution,
the Smagorinsky initial decay rate against the continuum solution, and the
initial diffusion-limited timestep for each. It writes both snapshots and
`metrics.json`. Pass `--athenak /path/to/reference-runs` to also check final
energies and the first two diffusion-limited timesteps against AthenaK.
That directory should contain `shear.hydro.hst` in each of `athena_uniform`,
`athena_smag`, `athena_uniform_limit`, and `athena_smag_limit`.

## Verify seed-1618 forcing

Run these commands from JathenAX using the AthenaK checkout and initialized
Kokkos submodule described above. The patch adds full-precision trace dumps
to a disposable copy of AthenaK.

```sh
ATHENAK=/path/to/athenak
JATHENAX=$PWD
CHECK_DIR=$(mktemp -d)
mkdir "$CHECK_DIR/source" "$CHECK_DIR/reference"
git -C "$ATHENAK" archive c50d3d5e962cf4c576de88c7b5dea33f4d2674dc | tar -x -C "$CHECK_DIR/source"
rmdir "$CHECK_DIR/source/kokkos"
ln -s "$ATHENAK/kokkos" "$CHECK_DIR/source/kokkos"
cp validation/athenak_forcing_trace.hpp "$CHECK_DIR/source/src/srcterms/"
(cd "$CHECK_DIR/source" && git apply "$JATHENAX/validation/athenak_forcing.patch")
cmake -S "$CHECK_DIR/source" -B "$CHECK_DIR/build" -DPROBLEM=turb -DCMAKE_BUILD_TYPE=Release
cmake --build "$CHECK_DIR/build" -j 8
(cd "$CHECK_DIR/reference" && ../build/src/athena -i "$JATHENAX/validation/seed1618.athinput")
JAX_PLATFORMS=cpu python -m validation.compare_forcing \
    "$CHECK_DIR/reference" "$CHECK_DIR/comparison"
```

The comparison checks the RNG stream, mode order and coefficients, raw and
normalized forcing, conserved state before and after each kick, and timestep
over 12 cycles. It replays AthenaK's timesteps and also runs with JathenAX
timestep selection. The trace spans three OU coefficient sets.

## Close a 3D AthenaK reference window

From the JathenAX root, use the same pinned AthenaK checkout and Kokkos
submodule as above. Build the float64 problem generator and use a fresh run
directory.

```sh
ATHENAK=/path/to/athenak
JATHENAX=$PWD
CHECK_DIR=$(mktemp -d)
PGEN=$(python -c 'import os, sys; print(os.path.relpath("validation/athenak_window", os.path.realpath(sys.argv[1] + "/src/pgen")))' "$ATHENAK")
cmake -S "$ATHENAK" -B "$CHECK_DIR/build" -DPROBLEM="$PGEN" -DCMAKE_BUILD_TYPE=Release
cmake --build "$CHECK_DIR/build" -j 8
mkdir "$CHECK_DIR/run"
(cd "$CHECK_DIR/run" && "$CHECK_DIR/build/src/athena" -i "$JATHENAX/validation/window_3d.athinput")
python -m validation.check_window "$CHECK_DIR/run" > "$CHECK_DIR/run/budget.json"
```

The deck starts from rest, drives a single periodic $32^3$ ideal-gas block,
and cools the fine thermal energy with $\tau_{\rm cool}=2$ in code units.
It records cycles 8 through 11 on aligned $8^3$ coarse cells. These settings
exercise face transport, forcing, and cooling during spin-up. Each measured
step must be free of EOS floor events.

Each `window.CYCLE.bin` starts with an ASCII line containing the three coarse
cell counts, the linear coarsening factor, the three coarse widths, the step
start time, its duration, the driving power per unit volume, the cooling time,
and $\gamma$. Native float64 arrays follow in `(field, z, y, x)` order.

| Field indices (inclusive) | Quantity and units |
|---|---|
| 0–6, 7–13 | Initial and final cell averages of $R,m_x,m_y,m_z,\overline E,U,K$. The fine kinetic moment determines $K$. |
| 14–18, 19–23, 24–28 | Fluxes of the five conserved variables through the low $x,y,z$ faces respectively, integrated in area and time and oriented along the positive axis. |
| 29–31, 32, 33 | Driving impulse, finite-kick work, and signed cooling contribution, each integrated over cell volume and the step. |

The checker uses the periodic neighbor's low face for each high face. Both
fluxes and cooling use the final SSPRK3 weights $(1/6,1/6,2/3)$. The driving
kick acts once after the final hydro stage. The report gives the largest
absolute cell residual for each conserved density, per step and across the
whole window. It also checks $\overline E=U+B+K$ and the imposed driving
power. The signed subgrid driving work sums each fine kick's work less the
kinetic energy change implied by its coarse momentum impulse.

For a cooling-only check, run the same executable in a fresh directory with
`turb_driving/dedt=0 problem/window_start_cycle=0 time/nlim=4` appended to
the input command. Run the checker on that directory. It verifies that
$R=1$, momentum and $K$ stay zero, and each step multiplies $U$ by
$1-h+h^2/2-h^3/6$, with $h=\delta t/\tau_{\rm cool}$.

## Measure the native stationary box

```sh
JAX_PLATFORMS=cpu python -m validation.run_stationary_box inputs/mach05_3d.toml
run=runs/stationary_mach05_jathenax/dedt1_32_seed1618_tcorr044
python -m validation.stationary_mach "$run/input.toml" "$run/history.csv" --interval 20 40 --blocks 6
```

This $32^3$ JathenAX run uses seed 1618, $c_{s,0}=3$, unit power per unit
volume, $\tau_{\rm cool}=14/3$, and a forcing update interval of 0.0025.
Set the OU e-folding time to one turnover at the peak forcing wavelength,
$t_{\rm eddy}=(L_{\rm box}/n_{\rm peak})/v_{\rm rms}$, using the stationary
3D mass-weighted turbulent speed. Here $L_{\rm box}=1$ and $n_{\rm peak}=2$.
The runner calls the native RK3 step and saves the final
snapshot, input deck, and per-cycle box means. Each step records the finite
kinetic work of its kick and the cooling loss integrated with the three RK
weights. The budget compares these with the independent change in total energy.
Measurements with boundaries inside a timestep use whole enclosed steps for
the discrete budget and time-interpolated endpoints for the statistical means.

Estimate Mach from the ratio of time-averaged turbulent kinetic and thermal
energies after subtracting center-of-mass motion. Compare the two half-window
means, six block values and energy autocorrelation times. The Mach uncertainty
is a paired block-jackknife standard error for this seed.

Across $20\le t\le40$ the first pilot, with $t_{\rm corr}=0.25$, gave
$\mathcal M_{\rm box}=0.49776\pm0.00169$ and $t_{\rm eddy}=0.44071$.
The revised deck sets $t_{\rm corr}=0.44$. Over the same interval it gave
$v_{\rm rms}=1.13675$, $t_{\rm eddy}=0.43985$, and
$\mathcal M_{\rm box}=0.49889\pm0.00232$. The two half-window Mach values
were 0.49723 and 0.50054. Mean driving, cooling, and total-energy storage
rates per unit volume were 1.000000, 1.001291, and $-0.001291$ on the
complete timesteps. The discrete budget closed to $1.1\times10^{-15}$ of
injected work. The applied spatial forcing pattern, after subtracting its
spatial mean and normalizing its rms amplitude at each sample, had a
measured $1/e$ lag of $0.467\pm0.016$ (six-block standard error).

### Measure spectral spin-up

The paired $32^3$ experiment uses the same seed-1618 driving and cooling in
two runs. The control starts at rest with $c_s=3$. In the other deck, set
`problem.initial_sound_speed = 2.28`, `problem.initial_vrms = 1.14`, and
`problem.initial_velocity_seed = 2718`. The new velocity has zero bulk
momentum and power in the solenoidal driving band $1\leq|\boldsymbol k|\leq3$.

Run `validation.run_stationary_box` on each deck with `--spectrum-dt 0.043985178860253216`.
Each `velocity_spectra.npz` stores the actual sample times and shell-integrated
power of the volume-mean-subtracted velocity. Sampling leaves the hydro CFL
schedule intact. The paired analysis takes its thresholds from the archived
`protocol.json` and reads the two histories in `warm/` and `control/`.

```sh
python -m validation.measure_spinup runs/stationary_mach05_jathenax/warmstart_spectrum_32
```

Using $t_{\rm eddy}=0.43985$, we averaged each diagnostic over one turnover
and required three consecutive passing intervals. Cooling had to match
injection within 2%. For the spectrum, we compared the absolute shell powers
at $k=1$–15 with the developed control. The 95th percentile of the control's
one-turnover RMS log deviations was 0.109.

With turbulent initial velocities, energy balance first passed over turnovers
1–2 and stayed within tolerance for all 89 complete intervals from turnover 1.
The spectrum first passed over 2–3, crossed the threshold again over 3–4,
and met the three-interval criterion over 4–7. Six of the next 83 spectra
exceeded the threshold. From rest at $c_s=3$, the spectrum met the same
criterion over 2–5 and energy balance over 35–38. The first three intervals
passing both tests spanned 4–7 in the turbulent start and 39–42 in the
control. Over $20\le t\le40$, the turbulent start gave
$\mathcal M_{\rm box}=0.5003\pm0.0021$ and cooling/injection of 0.99949.

### GPU timing and replay windows

On Apple silicon use the pinned Metal environment from the main README. The
stationary runner uses float32 hydro on Metal and accumulates its clock on
the host in float64. With a $128^3$ deck and a fresh output directory, pass
`--replay-at 24 30 36` to save three $64^3$ windows within the $t=20$–40
averaging interval. Each starts on the first fine step at or after its
requested time. The defaults are four consecutive fine steps and a 2:1 filter.
For the $32^3\to8^3$ training pilot, use `--replay-factor 4 --replay-steps 32`
to record a burst spanning several coarse CFL steps. Run
`python -m validation.check_window PATH/TO/window_XXXXXX` on each directory,
with the archived `input.toml` in its parent directory.

Each step stores the three RK stage states and the final state
$(R,\overline{\rho\boldsymbol u},\overline E,U,K)$, the six stage stresses
in $xx,xy,xz,yy,yz,zz$ order, the enthalpy and central third-moment fluxes,
and the RK-integrated low-face numerical fluxes. Face fluxes are per unit
area. Kick impulse, work, and positive cooling loss are per unit cell volume.
The pre-kick conserved state, OU update index and coefficients, force scale
and subtracted mean accompany each record. Stage arrays are in RK execution
order. Absolute time and step size are in the record header fields.

The $128^3$ M4 Max timing at $t_{\rm corr}=0.44$ measured 0.228 seconds per
warmed RK3 step from rest. A separate 192-step probe initialized from the
conservatively prolonged $32^3$ stationary snapshot measured a mean timestep
of 0.000642 over its final 128 steps. These timings and the $32^3$ CFL history
give about 1.9–2.0 hours to $t=20$ and 3.8–4.0 hours to $t=40$. Three
four-step spin-up windows passed the cell-by-cell budget check. Each used
356 MiB and took 4.55 seconds including evolution, filtering, transfer, and
disk writes, about 3.65 seconds beyond the ordinary steps. The spin-up and
replay jobs exited cleanly. The auxiliary developed-flow probe reproduced
its timestep twice and saved both results before a Metal client teardown
fault at process exit.

### Supervised stress fit

For the saved $32^3\to8^3$ window, run both commands from the repository root.
The reported fit used the M4 Max GPU with `JAX_PLATFORMS=METAL` and float32.

```sh
pilot=runs/tbnn_32_to_8_pilot
python -m validation.train_stress_tbnn "$pilot/window_015096" "$pilot/stress_fit_repeat"
python -m validation.verify_stress_tbnn "$pilot/window_015096" "$pilot/stress_fit_repeat"
```

The 114-parameter network takes $k/c_s^2$, $\mathrm{tr}(G)$, $S^{\rm dev}:S^{\rm dev}$,
and $\Omega:\Omega$ as inputs. Here $G=\Delta\nabla\widetilde{\boldsymbol v}/c_s$,
$S^{\rm dev}=\mathrm{dev}(\mathrm{sym}\,G)$, and $\Omega=(G-G^{\mathsf T})/2$.
Two coefficients multiply $S^{\rm dev}$ and $\mathrm{dev}(GG^{\mathsf T})$ as corrections
to a fitted eddy viscosity. All spatial derivatives use a centered periodic stencil.
The target is the deviatoric Favre stress. The objective weights the full tensor
Frobenius error by cell volume, fine timestep, and RK3 stage weight.

All fits and feature normalizations use the first 20 fine steps. Four steps separate
them from the final eight. On those eight steps the relative RMS errors were
0.976 for eddy viscosity ($C_\nu=0.07667$), 0.759 with two fitted constant
coefficients, and 0.735 for the TBNN. Mean deviatoric SGS production was 0.350
in the reference, 0.399 with eddy viscosity, and 0.346 with the TBNN. Production
was negative in 28.9%, 0%, and 12.5% of the weighted samples, respectively.
The full window spans 0.19 turnover times of one seed.
`stress_tbnn.npz` contains the weights, feature scaling, and model
conventions. `metrics.json` records both time blocks and the physical transfer.
The verifier runs inference from the checkpoint, compares every prediction with
the pre-save values, and checks realizability, cubic symmetry, a velocity boost,
the $K=0$ limit, and a directional finite difference of the weight gradient.

### Online stress training

Start from the supervised checkpoint above and the stationary seed-1618 fine
snapshot listed in `source.json`. Run either command from the repository root,
using a new output directory. Pass `--snapshot PATH` if the fine snapshot moved.

```sh
pilot=runs/tbnn_32_to_8_pilot
JAX_PLATFORMS=cpu python -m validation.train_online_tbnn "$pilot" "$pilot/online_cpu_repeat"
JAX_PLATFORMS=METAL ENABLE_PJRT_COMPATIBILITY=1 .venv-metal/bin/python \
    -m validation.train_online_tbnn "$pilot" "$pilot/online_metal_repeat"
```

The first update starts at $t=40$. The $8^3$ solver takes 32 RK3 steps to
apply the 32 archived driving kicks at their original times and scores the
end of every fourth step. The full duration is 6.85 nominal coarse CFL steps.
Each training trajectory starts once from filtered fine fields. Its subsequent
stress predictions use its evolving coarse state. Reference stress at all
three fine RK stages supplies a separate supervision term. The four equally
weighted trajectory fields are density, velocity, thermal energy density, and
$K$. Their squared errors use training-set fluctuation scales and elapsed-time
weights. The objective adds 0.01 times the normalized direct stress error.

The analytic and neural runs share the heat flux, SGS kinetic-energy flux and
decay models. Their coefficients and the loss scales use the same first 20
fine steps as the supervised fit. The decay coefficient comes from the
volume-integrated $K$ budget. A control inserts measured deviatoric stress
at each stage and keeps the same analytic terms. On the first trajectory its
normalized mean squared field error was 0.22821. The eddy-viscosity control
gave 0.25172, the pretrained TBNN gave 0.24245, and one Adam update at learning
rate 0.001 gave 0.24226. These scores include coarse discretization error.

The driver regenerates the archived update from the live $32^3$ box, compares
the loss, gradients, and updated weights, then continues the fine simulation
and Adam state for two more updates by default. New comparison times fall at
$40.09,40.10,\ldots,40.24$ with `--target-dt 0.01`. On the two new GPU
trajectories, respectively, the TBNN scored 0.24805 and 0.24137 before training
on those targets, and 0.24785 and 0.24111 after the update. Analytic stress
scored 0.25378 and 0.24091 on the same targets.

Each `reference_*.npz`
contains eight `target_time` values and `target_state` arrays in
$(R,m_x,m_y,m_z,\overline E,U,K)$ order, along with the finer stage and kick
records needed for replay. `first_trajectory.npz` holds the matched predictions
and reference. `stress_tbnn.npz` reloads with `jathenax.tbnn.load_weights`.
`metrics.json` records the fitted analytic terms, field errors, per-step
conservation residuals, finite differences, timestep refinement and checkpoint
reload checks. This pilot covers three consecutive trajectories from one seed.

### Frozen stress on a held-out seed

Use the training box's deck for the fine run, changing the driving seed to
1619 and the output path. Run to $t=40$ and use `validation.stationary_mach`
to check the history over $20\le t\le40$. The following commands assume the
new fine snapshot is at `$held/fine/final.npz`. On Apple GPUs use the pinned
Python environment with `JAX_PLATFORMS=METAL ENABLE_PJRT_COMPATIBILITY=1`.
The saved targets and weights can be scored directly from a checkout.

```sh
pilot=runs/tbnn_32_to_8_pilot
held="$pilot/heldout_seed1619"
fit="$pilot/online_metal/metrics.json"
python -m validation.evaluate_online_tbnn score "$held/reference" "$fit" \
    "$pilot/stress_fit/stress_tbnn.npz" "$held/short_analytic_repeat.json" --analytic
python -m validation.evaluate_online_tbnn score "$held/reference" "$fit" \
    "$pilot/stress_fit/stress_tbnn.npz" "$held/short_pretrained_repeat.json"
python -m validation.evaluate_online_tbnn score "$held/reference" "$fit" \
    "$pilot/online_metal/stress_tbnn.npz" "$held/short_online3_repeat.json"
python -m validation.evaluate_online_tbnn score "$held/reference" "$fit" \
    "$pilot/online50_metal/stress_tbnn.npz" "$held/short_online50_repeat.json"
```

To regenerate the reference or repeat the 50-update training, use a fresh
destination for each command.

```sh
python -m validation.evaluate_online_tbnn reference "$held/fine/final.npz" "$held/reference_repeat"
python -m validation.train_online_tbnn "$pilot" "$pilot/online50_metal_repeat" --live-updates 49
```

All models use the transport coefficients and loss scales fitted on seed 1618.
For a single eight-target replay, add `--blocks 1` to the `reference` command.
The snapshot can start between ticks once cooling has reached full strength.
The first target is the next multiple of 0.01, and its loss weight uses the
actual elapsed time from the snapshot.
The six short trajectories start at $40.00,40.48,\ldots,42.40$. Each is scored
against eight filtered fine states spaced by 0.01. The direct stress RMSE uses
the deviatoric stress at the start of each comparison interval and the full
tensor Frobenius norm.

For the long check, repeat each score command with `--continuous` and a new
output filename. Each model starts once at $t=40$ and advances to $42.48$
with the reference momentum impulses and direct SGS work. Accumulated sources
use the actual resolved work on the evolving coarse velocity and the RK3
cooling loss. The resulting box power can differ from the fine box's unit
power. The reported Mach number includes both resolved center-of-mass-frame
kinetic energy and $K$. Each output contains the sampled energies, a separate
resolved Mach number, stage minima and the energy-budget residual.

The longer run uses Adam at the original learning rate of 0.001. It accepts
updates that meet the physical and numerical checks and reduces the learning
rate on a failed check. Momentum accumulated from earlier trajectories can
raise the training loss on a particular new trajectory. The training report
records both losses and the current gradient dotted with the parameter update.

The seed-1619 fine run gave $\mathcal M=0.5000\pm0.0023$ and cooling/injection
of 1.00139 over $20\leq t\leq40$. The 50 updates on seed 1618 spanned
$40\leq t\leq44$ and 1,616 distinct coarse steps. Update 33 raised its
trajectory-plus-stress loss by $4.78\times10^{-6}$, with a positive gradient
dotted with the Adam step. The other 49 decreased their own loss. All 50 used
the original learning rate and the final checkpoint reproduced the trajectory
after reloading.

| Frozen stress on seed 1619 | Mean trajectory MSE | Density RMSE | Velocity RMSE | Thermal RMSE | SGS energy RMSE |
|---|---:|---:|---:|---:|---:|
| Analytic | 0.23414 | 0.5538 | 0.2329 | 0.5925 | 0.4739 |
| Supervised | 0.22736 | 0.5438 | 0.2287 | 0.5817 | 0.4723 |
| Three online updates | 0.22718 | 0.5436 | 0.2287 | 0.5815 | 0.4720 |
| 50 online updates | 0.22534 | 0.5414 | 0.2278 | 0.5791 | 0.4701 |

All four field RMSEs use the original training scales. The 50-update model
reduced trajectory MSE on all six starts relative to the supervised model.
Its relative direct stress RMSE rose from 0.7702 to 0.7779.

| Continuous $40\leq t\leq42.48$ | Mach | Mean resolved kinetic energy density | Mean $K$ | Driving power | Cooling power |
|---|---:|---:|---:|---:|---:|
| Filtered reference | 0.4937 | 0.4095 | 0.2242 | 1.0000 | 1.0029 |
| Analytic | 0.4277 | 0.2552 | 0.2368 | 1.0277 | 1.0375 |
| Supervised | 0.4314 | 0.2459 | 0.2520 | 1.0102 | 1.0318 |
| Three online updates | 0.4313 | 0.2464 | 0.2512 | 1.0102 | 1.0319 |
| 50 online updates | 0.4256 | 0.2674 | 0.2185 | 1.0104 | 1.0345 |

Each coarse model completed 5.65 fine-grid turnover times with positive
density, thermal energy, and $K$. For the 50-update model, energy storage
was $-0.02401$ per unit time and the residual against work minus cooling was
$1.8\times10^{-7}$. Halving the hydro timestep at identical kick times changed
its mean Mach by less than $10^{-6}$. The long trajectory shows a 35% deficit
in resolved kinetic energy even with mean $K$ within 3% of the reference.

### Fine-tuning the stress model at higher resolution

The $128^3$ training box uses seed 1618 and the same from-rest, five-turnover
spin-up as the seed-1619 reference. Filter it to $32^3$ and initialize the
stress model from the $8^3$ checkpoint after its 50 updates,
with fresh Adam moments at learning rate 0.001. All three models share that
run's transport coefficients, feature scaling and loss normalization.

```sh
small=runs/tbnn_32_to_8_pilot
large=runs/tbnn_128_to_32_pilot
fit="$small/online_metal/metrics.json"
initial="$small/online50_metal/stress_tbnn.npz"
export JAX_PLATFORMS=METAL ENABLE_PJRT_COMPATIBILITY=1
.venv-metal/bin/python -m validation.evaluate_online_tbnn score "$large/seed1619_reference" \
    "$fit" "$large/seed1618_online10/stress_tbnn.npz" "$large/seed1619_finetuned_repeat.json"
.venv-metal/bin/python -m validation.evaluate_online_tbnn score "$large/seed1619_reference" \
    "$fit" "$initial" "$large/seed1619_frozen8_repeat.json"
.venv-metal/bin/python -m validation.evaluate_online_tbnn score "$large/seed1619_reference" \
    "$fit" "$initial" "$large/seed1619_analytic_repeat.json" --analytic
```

The branch includes the exact seed-1619 targets and kicks used for these
scores. To repeat training from a fresh checkout, generate the seed-1618
fine state from the saved deck and start a new run.

```sh
if [ ! -f "$large/seed1618_fine/final.npz" ]; then
    .venv-metal/bin/python -m validation.run_stationary_box "$large/seed1618_fine/input.toml"
fi
.venv-metal/bin/python -m validation.finetune_online_tbnn "$large/seed1618_fine/final.npz" \
    "$initial" "$fit" "$large/seed1618_online10_repeat" --updates 10
```

Each update differentiates through the coarse RK3 trajectory using all
matched kicks. Only seed 1618 enters the optimizer. The fine simulation and
Adam state continue across updates. `metrics.json` records synchronized wall
times for reference evolution and filtering, explicit compilation of the main
kernels, gradient evaluation, the Adam step and post-update check, and writing
the compact replay record. The first and last updates include separate
directional finite-difference checks. The update timings exclude these checks
and the fine-box spin-up.

Ten updates used 80 targets from $t=2.20$ through $2.99$ on seed 1618.
The first three updates took 107.9, 87.1 and 95.9 seconds end to end on the
M4 Max GPU in float32. Fine evolution and filtering accounted for 80.8, 77.0
and 80.3 seconds. Explicit compilation took 4.2, 0.0 and 3.7 seconds.
The ten-update run took 15.8 minutes including both gradient checks. The
five-turnover spin-up took another 7.2 minutes. The reloaded checkpoint
reproduced the final training loss of 0.13655649 exactly.

All three frozen evaluations started from the same seed-1619 filtered state
at $t=2.199259$, replayed all 125 kicks and used the eight targets through
$t=2.27$. The trajectory score averages normalized mean-square errors in
density, velocity, thermal energy and subgrid kinetic energy. The stress
score is the relative Frobenius RMSE of the deviatoric stress.

| Stress model at $32^3$ | Trajectory MSE | Relative stress RMSE |
|---|---:|---:|
| Analytic eddy viscosity | 0.133970 | 0.97464 |
| Frozen $8^3$ training | 0.127975 | 0.73416 |
| Ten additional online updates at $32^3$ | 0.127452 | 0.73492 |

The checkpoint, per-update timings and paired scores are in
`runs/tbnn_128_to_32_pilot/seed1618_online10/` as `stress_tbnn.npz`,
`metrics.json` and `seed1619_comparison.json`.

## Calibrate the stationary Mach number

Use the same window executable and a fresh directory for each trial. The
$32^3$ calibration deck uses implicitly dissipative Euler hydro, a fixed
driving power per unit volume of 0.03703705, and a target Mach number of 0.5.
Set `tau_cool` and `rseed` in each saved input deck before running.

```sh
mkdir "$CHECK_DIR/pilot"
cp validation/stationary_3d.athinput "$CHECK_DIR/pilot/input.athinput"
(cd "$CHECK_DIR/pilot" && "$CHECK_DIR/build/src/athena" -i input.athinput > athena.stdout)
python -m validation.stationary_mach "$CHECK_DIR/pilot/input.athinput" \
    "$CHECK_DIR/pilot/stationary.hydro.hst" --interval 60 120 --blocks 6 \
    > "$CHECK_DIR/pilot/calibration.json"
```

The run writes full-precision box histories, an EOS event log, and a native
restart at the end. `window_start_cycle = -1` keeps fine-face recording off
during spin-up. To capture four developed steps, start the same executable
in a fresh directory with `-r` pointing to the final restart. Set
`problem/window_start_cycle` to its cycle, `time/nlim` to that cycle plus
four, and `time/tlim` beyond the burst. Run `validation.check_window` on
the resulting files.

The analysis subtracts bulk motion from kinetic energy and uses
the full kinetic energy to recover thermal energy from total energy.
It integrates both densities in physical time and then forms their ratio.
The reported Mach error is a paired, leave-one-time-block-out standard
error for a single seed. Check the block duration against the estimated
correlation times, compare the two half-interval means, and inspect the
floor counts. The long-history budget estimates cooling from sampled
endpoints. The short-window check uses the exact RK-weighted cooling.
Use the measured mean kinetic energy to choose the next cooling time and
repeat at the target resolution. For the $256^3$ reference, set all three
mesh and meshblock counts to 256 in the copied deck. A coarsening factor of
four then gives $\Delta=L/64$.

The $32^3$ runs gave the following measurements for $60\le t\le120$. The
error bars are one block standard error. The last column estimates the
cooling time for Mach 0.5 from the measured turbulent kinetic energy.

| $\tau_{\rm cool}$ | Seed | $\mathcal M_{\rm box}$ | Mean cooling / injected power | Estimated $\tau$ |
|---:|---:|---:|---:|---:|
| 7 | 1618 | $0.7020\pm0.0040$ | 0.99972 | 13.79 |
| 12 | 1618 | $0.5386\pm0.0024$ | 1.00033 | 13.93 |
| 14 | 1618 | $0.4974\pm0.0016$ | 1.00175 | 13.88 |
| 14 | 1619 | $0.4977\pm0.0016$ | 1.00260 | 13.91 |

At $\tau_{\rm cool}=14$, the kinetic and thermal correlation times ranged
from 1.35 to 1.72, compared with a block duration of 10. Restricting the
analysis to $80\le t\le120$ shifted either seed's Mach number by less than
0.0007. All four runs had zero reported EOS floor events. A four-step
window from the seed-1618 checkpoint at $t=120$ closed the total-energy
density budget to $7.27\times10^{-16}$ per coarse cell per step at
$\Delta=L/8$.

### Local CPU timing at the pilot settings

On an Apple M4 Max with 128 GiB of memory, the pinned float64 AthenaK
executable used twelve OpenMP threads and a single periodic block.
Both grid sizes used $\tau_{\rm cool}=14$, seed 1618, and a $64^3$ output
grid (coarsening factors two and four respectively).

| Fine grid | Measured seconds per RK3 step | Measured peak RSS | Estimated hours to $t=60$ | Estimated hours to $t=120$ |
|---|---:|---:|---:|---:|
| $128^3$ | 0.186 | 0.71 GiB | 1.6 | 3.1 |
| $256^3$ | 1.26 | 5.20 GiB | 21 | 42 |

The timing runs completed 256 and 16 steps, respectively. Per-step times
use interior cycles and exclude setup and final checkpoint I/O. Peak RSS
includes checkpoint output. The wall-time estimates scale the measured
$32^3$ cycle count at the same CFL with the fine grid's linear size.
Developed-flow wave speeds and sustained workstation load can change them.
A separate four-step $128^3\to64^3$ spin-up check wrote 272 MiB of
conservative budget data and closed each cell's update to float64 roundoff.
