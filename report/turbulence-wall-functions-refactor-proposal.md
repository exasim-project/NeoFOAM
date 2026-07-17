<!--
SPDX-License-Identifier: Unlicense
-->

# Turbulence wall functions — fix proposal, extra tests, and API refactor

**Branch:** `feat/turbNeoN`  ·  **Date:** 2026-07-17
**Companion to:** `report/steady-turbulence-wall-functions.md` (status + summary table)

This report is the *forward-looking* half: (A) proposals to close the remaining
kOmegaSST convergence-rate gap, (B) tests that would confirm each proposal, and
(C) a concrete refactor that pulls the wall-function machinery out of the ad-hoc
`pimple.cpp` bindings and into a readable Python API, with before/after snippets.

---

## Current state (what "green" rests on)

All 9 steady tests pass. Two facts frame the proposals below:

1. **Per-step discretisation is exact for every model** — at 2 SIMPLE iterations
   U/p match native to ~1e-12; kEpsilon and SpalartAllmaras are machine-precision
   throughout, kOmegaSST's near-wall k/ω carry ~1e-6.
2. **kOmegaSST converges to the same fixed point, slowly.** Its ω wall pin is
   ~1e5 (≈100× stiffer than epsilon's ~1e3). Per-cell relative diff vs native:

   | iterations | k | ω | ν_t |
   |---|---|---|---|
   | 30  | 4.6e-4 | 1.7e-4 | 1.6e-3 |
   | 500 | 4.8e-5 | 1.7e-4 | 6.3e-4 |

   k tightens ~10× from 30→500 iterations — contraction, not a different fixed
   point — but not to 1e-8 within the CI budget. The converged test therefore
   uses a peak-relative bound (`1e-3` of peak) for the stiff models. Closing this
   to machine precision is the open item.

---

## A. Proposed fixes (ranked by likelihood / cost)

### A1. Match native's linear-solver settings on the stiff equations *(cheap, most likely)*

Native solves ε/ω/k with OpenFOAM `PBiCGStab`+`DILU`; the framework maps these to
Ginkgo `Bicgstab`+`Ilu` (`map_fv_solution`). For a well-conditioned matrix the two
converge to the same answer, but the ω matrix after a `setValues` pin has a
condition number ~1e5, and Ginkgo/OpenFOAM can then stop at *different* residuals
for the same `tolerance`. The 2-iteration ω diff already sits at the near-wall
pinned cells.

- Set the ω solve to a **fixed, tight absolute tolerance** with a high `maxIter`
  and `relTol 0` (the case already does this — confirm Ginkgo honours the same
  `l1ScaledResidual` semantics as OpenFOAM; if not, that mismatch alone explains
  the residual gap).
- Try a **direct/near-direct** ω solve on this small case (Ginkgo LU) to remove
  the linear-solver variable entirely; if the converged diff collapses, the
  residual is linear-solver round-off, not the wall model.

### A2. Reduce the pin stiffness: eliminate the near-wall cell rather than "big-number" pin *(medium)*

`PDE::setConstraints` currently enforces ω via a fixed-value row. OpenFOAM's
`fvMatrix::setValues` does the *same* but then folds the pinned value into the
neighbours' source and zeroes the coupling **symmetrically**, so the reduced
matrix stays symmetric/positive-definite and the iterative solver behaves. If
`applyFixedValueConstraints` uses a diagonal-dominance ("big number") trick
instead of symmetric elimination, the conditioning differs.

- Audit `detail::applyFixedValueConstraints` (`src/datastructures/pde.cpp`) and,
  if it is a big-number pin, switch to **symmetric elimination** (move
  `a_ij · value` to RHS of row *i*, zero `a_ij`/`a_ji`). This is the single most
  principled fix and would also tighten the kEpsilon ε pin further.

### A3. Under-relax ω toward its wall value in the near-wall band *(cheap, diagnostic)*

Native applies equation relaxation (0.3) *before* the pin. We do too
(`solveImpl`: relax → pin). But the *post*-pin ω at the first unpinned cell
(cell 18 in pitzDaily) is where the diff concentrates. Temporarily raising the ω
relaxation to 0.5–0.7 and checking whether the converged diff shrinks tells us
whether the residual is relaxation-path-dependent.

### A4. Recompute `nearWallDist`/wallDist identically *(low, verify)*

Both sides use `meshWave`; confirm the NeoN `build_near_wall_dist` reads the same
`Foam::wallDist` object (it does) and that no `y` rounding differs. Low
probability but cheap to rule out with test B3.

---

## B. Additional tests to confirm the fix

### B1. Component test — pin the wall omega directly, assert equal to native

A unit-level test that constructs the pitzDaily mesh once and compares, **face by
face**, the NeoN `pin_omega_wall_cells` values against
`Foam::omegaWallFunctionFvPatchScalarField`'s `omega0` for a *fixed* input `k`.
This isolates the pin formula from the solve and the SIMPLE loop — if B1 is
bitwise but the converged solver still drifts, the residual is provably in the
linear solve (A1/A2), not the wall model.

```python
# test/operators/test_omega_wall_pin.py  (new)
def test_omega_wall_pin_matches_openfoam(pitzdaily_worker):
    k = seed_uniform("k", 0.375)
    neon_pin  = nfb.pin_omega_wall_cells(...)          # values view
    of_pin    = pyb.omegaWallFunction_omega0(k, ...)   # native omega0
    assert_allclose(neon_pin, of_pin, rtol=0, atol=1e-12)
```

### B2. Convergence-rate regression — assert monotone contraction

Rather than a single 500-iteration snapshot, sample the diff at
`[30, 100, 300, 500]` iterations and assert it is **monotonically decreasing** and
below an envelope. This encodes the real claim ("same fixed point, contracting")
and would catch a *divergence* regression that a single loose bound would miss.

```python
@pytest.mark.parametrize("n", [30, 100, 300, 500])
def test_komega_contracts(n, ...):
    diff = run_and_diff(model="kOmegaSST", end_time=n)
    assert diff["k"] <= CONTRACTION_ENVELOPE[n]   # e.g. {30:1e-3,100:5e-4,300:1e-4,500:5e-5}
```

### B3. `nearWallDist` equality test

Assert the NeoN `build_near_wall_dist` boundary values equal `Foam::wallDist.y()`
owner-cell values to 1e-14 (rules out A4).

### B4. Higher-iteration nightly parity

A `@pytest.mark.slow` variant at `end_time=3000` asserting kOmegaSST reaches
`rtol=1e-6`. Confirms "slow but converging" empirically; kept out of the fast CI
lane.

### B5. Second geometry (backward-facing step / channel with `yPlus < 11`)

pitzDaily wall cells sit in the log layer. A case with a **viscous-sublayer**
first cell (`yPlus < yPlusLam`) exercises the *other* branch of the omega/nutk
STEPWISE/BINOMIAL blenders and the Spalding `uTau` low-`yPlus` regime — currently
untested. This is the highest-value new *coverage* (independent of the A-fixes).

---

## C. Refactor + Python API improvement

The wall functions work, but the model-side half is five near-identical
boundary-face→cell scatter kernels bolted onto `pimple.cpp`, and each closure
open-codes which ones it calls. Two problems: (1) the scatter logic is duplicated
in C++ (`epsilon_wall_production`, `pin_epsilon_wall_cells`, `pin_omega_wall_cells`
all rebuild the same cornerWeight table), and (2) a closure reads like plumbing,
not physics.

### C1. One `WallFunction` object instead of five free bindings

**Before** — the closure wires raw bindings, threading `nearWallDist`, patch
names, and coefficients by hand (kOmegaSST `correct_omega` + `blend`):

```python
# blend(): near-wall k-production override — a bare binding call with 11 args
g_wall = nfb.epsilon_wall_production(
    g, omega, U, k, nu_vol, nut, komega_nearWallDist,
    neon_runtime, betaStar, 0.41, "omegaWallFunction",
)
...
# correct_omega(): pin then face-correct, two more bindings
nfb.pin_omega_wall_cells(eqn, omega, k, nu_vol, komega_nearWallDist,
                         neon_runtime, beta1, betaStar, 0.41)
eqn.solve()
nfb.correct_scalar_bc_ctx(omega, k, nu_vol, komega_nearWallDist)
```

**After** — a `WallFunction` resolved from the field's registered BC carries the
patch, the coefficients, and the scatter; the closure states intent:

```python
# build(): one model, resolved from the omega field's on-disk BC
wf = WallFunction.for_field(omega, k=k, nu=nu_vol, U=U)   # knows it's omegaWallFunction

# blend(): the k production the wall function feeds
g_wall = wf.production_override(g)          # zeroes wall cells, adds log-law G

# correct_omega():
wf.pin(eqn)                                 # near-wall cell pin (into the matrix)
eqn.solve()
wf.correct_faces(omega)                     # BoundaryContext face correction
```

`WallFunction.for_field` inspects the registered BC name (`omegaWallFunction` /
`epsilonWallFunction` / `nutUSpaldingWallFunction`) and selects the right pin
formula and context keys — no `wall_patch` string, no `cmu=betaStar` passed at the
call site, no `nearWallDist` threading. Non-wall-function fields return a no-op
`WallFunction`, so the same three lines work unchanged in the no-WF turbulentBox
case (today that relies on every binding silently no-op-ing).

### C2. Collapse the C++ duplication behind one scatter primitive

**Before** — `epsilon_wall_production`, `pin_epsilon_wall_cells`,
`pin_omega_wall_cells` each contain the identical ~30-line cornerWeight build +
per-patch `parallelFor` scatter, differing only in the per-face expression.

**After** — one templated helper in `include/NeoFOAM/…/wallFunctionScatter.hpp`:

```cpp
// cornerWeight-averaged scatter of a per-face functor into a per-cell field.
template <class FaceExpr>
void wallScatter(const fvcc::VolumeField<scalar>& drivenBy,   // holds the WF patch
                 const std::string& wallPatch,
                 const nf::RunTime& rt,
                 NeoN::Vector<scalar>& out,                    // accumulate here
                 FaceExpr expr);                               // (facei, owner) -> scalar
```

so the production override and both pins become three-line callers:

```cpp
// pin_omega  == wallScatter(omega, "omegaWallFunction", rt, values,
//                           KOKKOS_LAMBDA(i, c){ return cw[c]*blendedOmega(i,c); });
```

The cornerWeight table is built once and cached (as the reference C++
`kOmegaSST.cpp` already does with `cornerWeightsBuilt_`), removing the per-call
rebuild that today runs every SIMPLE iteration.

### C3. Move the pin/override into the closure DSL (stretch)

The last C++ leak in an otherwise all-Python closure is the boundary-face→cell
scatter. A DSL surface would keep the whole closure in Python:

**Before** (imperative, C++ binding):

```python
nfb.pin_omega_wall_cells(eqn, omega, k, nu_vol, y, rt, beta1, betaStar, 0.41)
```

**After** (declarative, part of the equation):

```python
eqn = nfb.PDESolverScalar(
    nn.imp.ddt(omega) + nn.imp.div(phi, omega) - nn.imp.laplacian(d_omega, omega)
    - nn.exp.source(prod) + nn.imp.source(sp, omega) + nn.imp.susp(susp, omega)
    + nn.imp.wall_pin(omega, wf.omega_pin_values()),      # constraint as a term
    omega, rt,
)
eqn.solve()
```

`imp.wall_pin` compiles to the same `setConstraints` call but reads as part of the
equation, matching how `imp.source`/`imp.susp` already fold matrix manipulations
into the expression.

### C4. Net effect on the closure

kOmegaSST `correct_omega` today is ~20 lines of solve + 3 binding calls with
coefficient-threading. After C1+C3 it is the transport equation plus two
intention-revealing lines (`wf.production_override`, `imp.wall_pin` /
`wf.correct_faces`). The five `pimple.cpp` bindings collapse to one scatter
primitive + one `WallFunction` binding, and adding a new wall-function family
(e.g. `nutLowReWallFunction`) becomes "register the BC + add a pin formula",
not "write another 30-line scatter binding and thread it through the closure".

---

## Recommended order

1. **A2 audit** (symmetric-elimination pin) — the principled root-cause fix; pairs
   with **B1** (component pin test) to prove it.
2. **A1** (linear-solver settings) — cheap, and **B2** (contraction regression)
   confirms whichever of A1/A2 lands.
3. **B5** (viscous-sublayer geometry) — coverage that is valuable regardless of
   the convergence fix.
4. **C1/C2** refactor once the physics is locked (don't refactor a moving target),
   then **C3** as the API polish.
