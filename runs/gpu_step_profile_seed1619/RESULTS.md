# GPU timestep profile for driven JathenAX turbulence

These measurements preceded the driving and face-flux optimizations. The subsequent
whole-step benchmarks are in the [solver timing table](../solver_optimization_seed1619/timings.csv).

The largest cost at 128³ and 64³ is PPM4 reconstruction. At 32³ it is rebuilding the Fourier driving field. Measurements use the Apple M4 Max GPU, JAX 0.4.34 with Metal, FP32, and seed 1619. The workload is the periodic 3D ideal-gas pilot with PPM4, Roe, three-stage Runge–Kutta, adaptive CFL, normalized driving, cooling, and per-step diagnostics.

## Where a full timestep goes

| Rank at 128³ | Operation | 128³ | 64³ | 32³ |
|---:|---|---:|---:|---:|
| 1 | PPM4 reconstruction | 46.3% | 40.6% | 19.9% |
| 2 | Euler face fluxes and Roe Riemann solve | 40.1% | 31.4% | 22.5% |
| 3 | RK updates, recovery of primitives, cooling, normalized forcing kick | 5.9% | 5.4% | 7.8% |
| 4 | Refresh Fourier driving field | 3.8% | 14.4% | 39.2% |
| 5 | Flux divergence | 3.2% | 4.0% | 0.9% |
| 6 | CFL, pilot diagnostics, and cached-force index check | 0.75% | 4.2% | 9.8% |

Each measurement starts at the saved state after five eddy times. The compiled pilot was timed after warm-up, with GPU synchronization, in 16 randomized rounds of two calls per variant. To isolate each hydro operation, the benchmark supplies the exact face states, fluxes, or divergences from the corresponding direction and RK stage. Each substitution produced the same conserved state as the full step. The time differences estimate the cost removed in that compiled calculation. Compiler fusion and intermediate-array reads can shift the split between adjacent operations. [Times, shares, and paired bootstrap intervals](attribution.csv)

The driving field refreshed on 24.4%, 50.7%, and 100% of late steps at 128³, 64³, and 32³ respectively. The table weights refresh and reuse steps by those frequencies. Independently, pairing each refresh with nearby reuse steps in the original simulation gives a driving-field cost of 4.4% at 128³ and 13.9% at 64³. At 128³ the short profile averaged 225 ms per step, compared with 318 ms over the second half of the full simulation. In the full 128³ run, timed stepping accounted for 1,042.49 of approximately 1,043.51 seconds from log creation to process shutdown.

## Changes timed in the full pilot

| Benchmark-only change | Whole-step speedup at 128³ | 64³ | 32³ |
|---|---:|---:|---:|
| Compute each bounded PPM4 edge once using slices of a periodically padded array | 1.57× | 1.50× | 1.17× |
| Keep Roe flux components separate until assembling the result | 1.06× | 1.01× | 0.91× |
| Sum Fourier modes as a matrix multiply with precomputed transverse phases | 1.04× | 1.17× | 1.81× |
| All three changes together | 1.84× | 1.99× | 1.75× |

The PPM4 prototype retains the cell-wise limiter. Changing the order of the Fourier sum produced a relative RMS difference of about $3\times 10^{-7}$ in the refreshed raw acceleration field at each size. The one-step diagnostic values agreed within FP32 tolerance. These prototypes live with the measurements. [Paired full-step timings](prototype_timings.csv) · [Prototype implementations](candidates.py)

## Consecutive adaptive steps

Each variant also ran 24 consecutive steps, with the state, CFL timestep, forcing cache, and physical time advancing normally. Timings are means over five interleaved repetitions and include the transfer of the pilot diagnostics on every step.

| Grid | Original ms/step | Shared PPM edges | Fourier matrix multiply | All three changes |
|---:|---:|---:|---:|---:|
| 128³ | 238.82 | 153.52 (1.56×) | 232.32 (1.03×) | 128.84 (1.85×) |
| 64³ | 22.53 | 14.92 (1.51×) | 19.17 (1.18×) | 11.14 (2.02×) |
| 32³ | 10.16 | 8.64 (1.18×) | 6.24 (1.63×) | 6.24 (1.63×) |

PPM alone produced bitwise-identical final conserved fields and identical elapsed simulation times on all three grids. With Fourier summation reordered, the final conserved fields differed by $1.3\times10^{-7}$ in relative RMS on each grid. All variants had finite diagnostics at every step and positive stage densities and internal energies above the configured floors. [128³ results](n128_trajectory_meta.json) · [64³ results](n64_trajectory_meta.json) · [32³ results](n32_trajectory_meta.json)

## Repeating the measurement

Run from the repository root with the same JathenAX source. Supply a completed pilot directory containing `input.toml`, `history.csv`, and `final.npz`. On a CUDA system, copy [the profiler](profile_step.py) and [the prototypes](candidates.py) together. The harness enables FP64 on CUDA. It retains its input buffers for repeated measurements. Metal already disables buffer donation. On the H200, compare the result with the native production run using its usual buffer donation, grid size, and diagnostics.

```sh
PYTHONPATH=. JAX_PLATFORMS=cuda python runs/gpu_step_profile_seed1619/profile_step.py /path/to/pilot --output /path/to/profile
```

Add `--candidates` for the single-step prototype comparison, or `--trajectory 24 --rounds 5` for consecutive adaptive steps. The measurements here cover forward hydrodynamics. Online TBNN training, gradient calculations, and replay output were inactive.
