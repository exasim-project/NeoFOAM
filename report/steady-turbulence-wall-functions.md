<!--
SPDX-License-Identifier: Unlicense
-->

# Steady (SIMPLE) wall-function parity for the NeoN turbulence closures

**Branch:** `feat/turbNeoN`  ·  **Date:** 2026-07-17

## TL;DR

`incompressibleFluidNeoN` now runs the pure-Python NeoN `kEpsilon`,
`kOmegaSST` and `SpalartAllmaras` closures **with wall functions** in steady
SIMPLE, and matches the trusted native OpenFOAM solver (`incompressibleFluid`)
on `pitzDailySteady`. The parameterised test
`test/solver/incompressibleFluidNeoN/test_steady_vs_incompressibleFluid.py`
covers all three models × three horizons (2-iteration round-off, 500-iteration
converged, native `simpleFoam` reference) — **9 tests, all green**.

The wall functions were wired **without modifying the `src/NeoN` submodule**:
the boundary-condition classes live in `include/NeoFOAM/…` and self-register into
libNeoN's runtime-selection table from the top-level bindings via CRTP
(`fieldFactories.cpp`); the model-side halves (near-wall `G` override, near-wall
cell pins, context-aware BC correction) are small top-level bindings in
`pimple.cpp`; the closures themselves stay readable Python.

## Result summary

Per-cell max relative difference vs native OpenFOAM (`peak` = peak |reference|).

| Model | Wall functions | 2-iter (per-step) | 500-iter (converged) | vs native `simpleFoam` |
|---|---|---|---|---|
| **kEpsilon** | epsilon / kqR / nutk | machine precision (U 6e-11, ε 1.4e-9, ν_t 2e-14) | machine precision (ε 3.4e-7 / peak 5.4e3 ≈ 6e-11) | bitwise (0.0) |
| **SpalartAllmaras** | nutUSpalding (nuTilda fixedValue 0) | machine precision (U 3.5e-11, ñ 5e-16, ν_t 5e-16) | ~1e-4 (U 2.9e-5, ñ 8e-5) | bitwise (0.0) |
| **kOmegaSST** | omega / kqR / nutk | U/p 1e-12, ν_t 2e-7, k/ω ~1e-6 | ~1e-3 (U/p 1.8e-5, k 4.8e-5, ν_t 6e-4) | bitwise (0.0) |

Test tolerances (in `MODELS`): kEpsilon strict `rtol=1e-8` at both horizons;
SpalartAllmaras strict per-step, `1e-3`-of-peak converged; kOmegaSST `rtol=3e-5`
per-step, `1e-3`-of-peak converged.

## What each wall function needed

The wall functions are "half-in-BC, half-in-model": the registered BC sets only
the wall **face** value (via the 2-argument `correctBoundaryCondition(field,
BoundaryContext&)` — the no-arg overload is a no-op); the turbulence **model**
must supply the near-wall production override and the near-wall cell pin.

| Piece | kEpsilon | kOmegaSST | SpalartAllmaras |
|---|---|---|---|
| Face-value BC | epsilon / kqR / nutk | omega / kqR / nutk | nutUSpalding |
| BC context | `(k, ν, y)` via `correct_scalar_bc_ctx` | `(k, ν, y)` | `(U, ν, y)` via `correct_scalar_bc_ctx_u` (new) |
| Near-wall `G` override (k eqn) | `epsilon_wall_production` | same binding, `wall_patch="omegaWallFunction"`, `Cmu=βStar` | — (no k equation) |
| Near-wall cell pin | `pin_epsilon_wall_cells` | `pin_omega_wall_cells` (new) | — (nuTilda is fixedValue 0) |

`nearWallDist` (boundary faces = owner-cell wall distance) is built once per model
(`build_near_wall_dist`) as the wall-function input `y`.

## Three fixes that were decisive

1. **kOmegaSST production cap ordering.** `kOmegaSSTBase::Pk(G) = min(G,
   c1·βStar·k·ω)` is evaluated **after** the omega solve, so the near-wall
   log-law `G` (set at `updateCoeffs`) is capped against the **pinned** omega.
   At high-shear wall cells (the inlet/wall corner) the raw log-law `G`
   overshoots and the cap binds — kEpsilon has no such cap. Fix: `blend`
   publishes the wall-overridden **but uncapped** `G`; `correct_k` applies the
   `min(G, c1·βStar·k·ω)` cap with the post-solve omega. This alone moved the
   corner cells from 80% off to machine precision.

2. **nutUSpalding tolerance.** OpenFOAM's `nutUSpaldingWallFunction` defaults to
   `maxIter=10, tolerance=0.01`, and at the default tolerance it **skips** the
   restart-preservation branch. NeoN was using `tolerance=1e-9` and *always*
   preserving — converging to a slightly different `uTau`, which drifted the
   momentum solve by iteration 2. Fix (`include/NeoFOAM/…/nutWallFunction.hpp`):
   `TOLERANCE=0.01`, drop the restart branch → SA matched to machine precision
   per-step.

3. **omega BINOMIAL blender = STEPWISE for epsilon/nutk** (carried over from the
   kEpsilon work): OpenFOAM v2406 defaults epsilon/nutk to STEPWISE but omega to
   BINOMIAL; the NeoFOAM headers now match each.

## Next step: refactor

The wall-function plumbing works but is deliberately staged for a cleanup pass:

1. **Dissolve the `pimple.cpp` wall-function bindings into a proper NeoFOAM
   wall-function model.** `epsilon_wall_production`, `pin_epsilon_wall_cells`,
   `pin_omega_wall_cells`, `correct_scalar_bc_ctx{,_u}` and
   `build_near_wall_dist` are five near-identical boundary-face→cell scatter
   kernels bolted onto the bindings. They should collapse into one reusable
   `WallFunctionCorrection` abstraction (cornerWeight build + per-patch scatter)
   that the BC classes and the models share, so a closure declares *which* wall
   treatment it uses rather than open-coding the scatter.

2. **Move the model-side override into the closure DSL.** The near-wall `G`
   override and cell pin are the only parts of the closures that drop out of the
   readable Python field maths into C++ scatter. A small `imp.wall_pin(...)` /
   `wall_production(...)` DSL surface would keep the whole closure in Python.

3. **kOmegaSST convergence rate.** kOmegaSST (omega pinned to ~1e5, ~100×
   stiffer than epsilon's ~1e3) contracts the same fixed point far more slowly
   than kEpsilon — at 500 iterations it is at ~1e-3 of peak, not machine
   precision (per-step discretisation *is* exact: U/p ~1e-12). The converged
   test therefore uses a peak-relative bound for the stiff models. Worth
   investigating whether matching native's linear-solver settings / a tighter
   omega relaxation recovers machine-precision convergence within a feasible
   iteration budget; until then the 2-iteration per-step test is the sharp
   discretisation check.

4. **Register the BCs in one place.** `fieldFactories.cpp` ODR-uses the
   `Register<>::REGISTERED` statics to force self-registration. This is fine but
   implicit; a single explicit `registerWallFunctions()` entry point would be
   clearer and greppable.

## Files touched (no `src/NeoN` submodule changes)

- `include/NeoFOAM/fvcc/boundary/volume/nutWallFunction.hpp` — nutUSpalding
  tolerance = 0.01, drop restart branch.
- `src/bindings/fieldFactories.cpp` — register omega + nutUSpalding BCs.
- `src/bindings/pimple.cpp` — `correct_scalar_bc_ctx_u`, `pin_omega_wall_cells`,
  `epsilon_wall_production` generalised with a `wall_patch` argument.
- `src/neofoam/turbulence/models/kOmegaSST.py`,
  `src/neofoam/turbulence/models/spalartAllmaras.py` — wire the wall functions.
- `test/solver/incompressibleFluidNeoN/test_steady_vs_incompressibleFluid.py` —
  parameterise over the three models; `cases/pitzDailySteady/models/<name>/`
  overlays.
