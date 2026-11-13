# JathenAX validation

Run from the repository root after installing JathenAX. Install pytest for
the unit tests.

```sh
python -m pip install pytest
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python -m pytest -q
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
