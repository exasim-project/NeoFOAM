# Report — Turbulence as a pure-Python ModelSpec on NeoN

**Branch:** `feat/turbNeoN`  ·  **Date:** 2026-07-11

## Goal

Simplify how turbulence models are written and used: express them as **pure-Python
ModelSpec** models running on the **NeoN** backend (define `nut` / `nuEff`, and any
transport equations, in Python), instead of the C++ turbulence factory
(`nfb.create_turbulence_model`). Validate each model against **pybFoam**
(OpenFOAM) as the trusted reference.

Simplicity is the objective: a new closure should be a small, readable Python file
plus a case, not a C++ model + nanobind binding + rebuild.

**Hard requirements:**

- **Runtime-selectable.** The active model is chosen at run time, not hard-coded —
  a single selection entry dispatches to the right ModelSpec (or the C++ fallback),
  the NeoN analogue of the pybFoam `select_turbulence_model`.
- **Driven by the OpenFOAM turbulence dict.** Selection reads the standard
  `constant/turbulenceProperties` (`simulationType laminar|RAS|LES`, with
  `RAS { RASModel <name>; }` / `LES { LESModel <name>; }`) — no new config format.
  The same on-disk case that runs OpenFOAM selects the pure-Python NeoN model.
- **No regressions.** A model without a pure-Python implementation still resolves to
  the existing C++/OpenFOAM model through the same entry, so every case keeps running.
- **Verified by the test.** The parity test itself reads the model from the case's
  `constant/turbulenceProperties` and asserts the selection — dict-driven runtime
  selection is part of what passes, not an assumption.

Current state against these: selection already reads `constant/turbulenceProperties`
via `TurbulencePropertiesConfig` + `selection.model_name` and dispatches in
`build_neon_turbulence` (pure-Python `laminar`, C++ fallback otherwise) — see status
below. What remains is broadening the pure-Python side (gated by the missing
primitives in *What is missing*).

---

## Result (2026-07-11): laminar, kEpsilon **and** SpalartAllmaras parity to machine precision

All three bundled pure-Python NeoN models now match pybFoam **after one `correct`
step**, cell-by-cell, on every field each model owns:

| Model | field | max abs diff | peak | notes |
|---|---|---|---|---|
| `laminar`         | `nut`     | `0.000e+00` | `1.0`    | exact (`nut = 0` both sides) |
| `kEpsilon`        | `nut`     | `3.9e-18`   | `3.6e-3` | ~1e-15 relative — bit-identical |
| `kEpsilon`        | `k`       | `2.2e-16`   | `0.60`   | transport unknown matches too |
| `kEpsilon`        | `epsilon` | `7.1e-15`   | `19.9`   | |
| `SpalartAllmaras` | `nut`     | `8.7e-19`   | `2.0e-3` | ~4e-16 relative |
| `SpalartAllmaras` | `nuTilda` | `8.7e-19`   | `2.0e-3` | transport unknown |

`SpalartAllmaras` (`neon_spalartAllmaras.py`) is the first closure needing the gaps
identified in `neon-turbulence-komegasst-spalartallmaras-gaps.md`: it is authored as
**readable NumPy on the host** (`chi`, `fv1`, `fv2`, `Stilda`, `r`, `g`, `fw`) with
four new small bindings — `copy_from_host` (host→field write, the key one),
`read_wall_distance` (`y` = `Foam::wallDist`), `vorticity_magnitude`
(`sqrt(2) mag(skew gradU)`) and `mag_sqr_grad` (`magSqr(grad nuTilda)`). `ft2`
defaults off (drops out) and RAS `dTilda = y`, so the model matches
`Foam::RASModels::SpalartAllmaras::correct` term for term. Its parity case adds one
`wall`-type patch (so `y` is defined) but no wall functions.

The `kEpsilon` step genuinely exercises **convection, variable-coefficient
diffusion, production, dissipation, bounding and `correctNut`**: the case seeds
**random per-cell** `k`/`epsilon` and a **random divergence-free** velocity
`U = (a y + b z, c x + d z, e x + f y)` (each component independent of its own
coordinate ⇒ `div U = 0` exactly, yet full off-diagonal `grad U` ⇒ nonzero
production). Because `div U ≈ 2.7e-15`, OpenFOAM's dilatation `SuSp(divU, …)` terms
vanish **identically on both sides** rather than being dropped only on the NeoN
side — so the NeoN closure matches `Foam::RASModels::kEpsilon::correct` term for term
(no wall functions: a fresh no-wall box case, `test/turbulence/parity_base/`).

### Test architecture — one `Foam::Time` per process

Several `Foam::Time` objects in one process corrupt the shared OpenFOAM registry and
segfault, so `test_neon_turbulence_parity.py` orchestrates three **subprocess**
roles (`_parity_worker.py`): `setup` (blockMesh + seed identical random fields),
`reference` (pybFoam), `subject` (NeoN). Each writes a `.npy`; the parent only
compares. Adding a model = drop a `turbulenceProperties` under `parity_models/<name>/`
and list the name in `CASES`.

### Gotchas that made the NeoN scalar transport solve work (all required)

1. **argList + `Foam::Time` lifetime** — build the NeoN `RunTime` from
   `pyf.Time(pyf.argList([...]))` and keep **both** Python-alive; `Foam::Time` holds a
   raw reference to the argList, and the two-arg `Time(rootPath, case)` constructor
   doesn't wire the dict environment `solve()` reads (→ segfault deep in
   `polyMesh::dbDir` / the linear solve).
2. **`nn.rotate_old_times(field)`** before assembling — `imp.ddt` (BDF1) reads the
   field's previous-time entry from the database; without it, assembly throws
   `unordered_map::at`.
3. **Operand lifetime** — the NeoN implicit operators hold *references* to their
   operand fields, so every inline coefficient (`nut/sigma + nu`, `C2*eps/k`, …) must
   be bound to a local that outlives `solve()`, else it's GC'd and read as freed memory.
4. **Literal (non-regex) fvSolution solver keys** — `PDESolverScalar` looks the
   field's solver up by a literal `unordered_map::at("epsilon")`; a regex key like
   `"(k|epsilon)"` resolves for pybFoam but misses on NeoN.
5. **`nn.field_max`** (not `nfb.field_max`) for the `bound()` floors.
6. **Hard `os._exit(0)`** at the end of each worker — NeoN/Kokkos teardown at normal
   interpreter exit segfaults *after* the result is already on disk.

---

## Current status

### Working

| Piece | File | State |
|---|---|---|
| Parity test (extensible, parametrized) | `test/turbulence/test_neon_turbulence_parity.py` | ✅ green |
| Runtime-selectable registry + wrapper + selector | `src/neofoam/turbulence/neon.py` (`turbModels`, `turb_model`, `NeoNTurbulence`, `build_neon_turbulence`) | ✅ |
| **Pure-Python laminar** model | `src/neofoam/turbulence/models/neon_laminar.py` (`NeoNLaminarTurbulence`) | ✅ reproduces C++ exactly |

- **Registry (`turbModels`)** — `neon.py` holds a name → factory registry; each
  pure-Python model lives under `src/neofoam/turbulence/models/` and registers via
  the `@turb_model("<name>")` decorator (import side-effect). `build_neon_turbulence`
  resolves the dict name against `turbModels`, falling back to the NeoN C++ model.
- **One example case + diff** — a single base case (`test/setup_saddes`, all fields
  on disk: `U`/`p`/`k`/`epsilon`/`nut`/`nuTilda`) is shared by every model; a model
  is selected by applying its diff — a `constant/turbulenceProperties` — over the
  base, and reads (or generates) the fields it needs.
- **Test structure** — a `MODELS` registry drives one generic test; adding a model
  = one entry (its turbulenceProperties diff + the fields to check). Runs entirely
  **in-process** (in-process blockMesh, no subprocesses, no code-as-strings); the
  pybFoam reference and the NeoN subject read the same on-disk `U`.
- **Validated results** (NeoN vs pybFoam, single time state):
  - `laminar` — no eddy viscosity (`nut` is `None` by definition); the build/validate
    flow is exercised.
  - `SpalartAllmarasDDES` — `nut` matches pybFoam **exactly** (max abs diff `0.0`).
- **Pure-Python laminar** (`NeoNLaminarTurbulence`) — computes `nut = 0` and
  `nuEff = surfaceInterpolate(nu)` from NeoN Python primitives (no C++ turbulence
  factory). Verified against the C++ laminar model: `nuEff` diff `0.0`, `nut` diff
  `0.0`. It owns **no** `grad(U)` — that is kinematic, produced by the momentum
  predictor, only *consumed* by the viscous stress.

### Design that emerged

- A ModelSpec turbulence model provides `nut` / `nuEff` and, for a closure with a
  transport equation, a **`correct` operation** (solved once per step after the
  PIMPLE loop). `laminar` declares no fields and no operation; a RAS/LES closure
  adds its transport fields and a `correct` solve.
- The comparison is field-level and extensible: what is verified per model can grow
  (`nuEff`, `k`, `omega`, `nuTilda`, …) as the bindings expose those fields.

---

## What is missing

Two independent gaps: **(A)** binding surface needed to *write* non-trivial closures
in Python, and **(B)** field access needed to *validate* more quantities.

### A. NeoN Python bindings missing to author a closure in Python

A closure like SA-DDES (or `kEpsilon`/`kOmegaSST`) is mostly **algebra on fields**
plus a transport PDE. The transport PDE is expressible today
(`nn.imp.ddt/div/laplacian` + `nfb.PDESolverScalar`, as the pressure solve already
does), but the algebra is not:

| Needed for a closure | Bound today? | Blocker |
|---|---|---|
| Field arithmetic `a*b`, `a+b`, `a/b`, `a**2` on `ScalarVolumeField` | ❌ | `ScalarVolumeField` exposes **no** `__add__/__mul__/__truediv__/__pow__` |
| Scalar math `sqrt`, `pow`, `tanh`, `min`, `max`, `exp` on fields | ❌ | `nn` exposes only `dot`, `mag`, `scalar_mag` |
| Strain/vorticity from `grad(U)` | ❌ | `TensorVolumeField` has **no** Python accessors; no `symm`/`skew`/`mag(tensor)` |
| Wall distance `d` (`patchDist`/`meshWave`) | ❌ | no binding |
| Evaluate an operator to a field (`divDevReff(U)` as a field) | ❌ | operators only assemble into a `PDESolver`; `assemble()` returns nothing, `LinearSystemVector` has only `copy_to_host`/`reset` (no matrix/spmv/residual) |

Consequence: **SA-DDES cannot be written in pure Python today** — its damping
functions (`fv1`, `fv2`, `fw`, `ft2`), the DDES length scale / shielding, the
production (needs vorticity) and destruction (needs wall distance) terms have no
Python surface. Implementing it would mean exposing essentially the whole
turbulence-math layer, i.e. rebuilding the C++ model in Python for no benefit.

### B. Field access missing to validate more quantities

- `nuEff` on NeoN is a **surface** field; pybFoam's `nuEff` is a **volume** field —
  a fair comparison needs interpolation (or a volume `nuEff` accessor).
- `k` / `omega` / `epsilon` / `nuTilda` are not exposed by the NeoN
  `TurbulenceModel` binding (only `nut` / `nu_eff` / `grad_u`), so they can't yet be
  validated field-by-field.
- `TensorVolumeField` has no host copy, so `grad(U)` can't be diffed directly.

---

## Feasibility by model tier

| Tier | Pure-Python ModelSpec? | Notes |
|---|---|---|
| `laminar` (no eddy viscosity) | ✅ done | matches C++ exactly |
| Algebraic / one-equation with a transport PDE (`kEpsilon`, …) | ⚠️ partial | transport solve is expressible; **source-term algebra + math bindings are missing** |
| SA-DDES (DDES shielding, wall distance, vorticity, damping) | ❌ not now | needs the full field-algebra/math/wall-distance/tensor surface |

---

## Recommendation / next steps (to decide)

1. **Ship the Python laminar ModelSpec + the extensible parity harness now.** They
   are the simplification win: laminar is authored in Python and validated; new
   *simple* models drop into the registry. Keep SA-DDES via the C++ factory (wrapped
   by the same selection) until bindings exist.
2. **To unlock pure-Python closures, add a small, reusable NeoN Python surface**
   (this is the high-leverage work, independent of any one model):
   - field arithmetic operators on `ScalarVolumeField` (`+ - * /`, scalar `**`);
   - elementwise math (`sqrt`, `pow`, `tanh`, `min`, `max`, `exp`);
   - `symm`/`skew`/`mag` on `TensorVolumeField` + a host accessor;
   - wall distance (`patchDist`).
   With these, `kEpsilon`/`kOmegaSST` become a readable Python file; SA-DDES becomes
   feasible.
3. **Improve validation coverage** (cheaper): expose `k`/`omega`/`nuTilda` on the
   NeoN `TurbulenceModel` handle and a volume `nuEff`, so more fields are checked
   against pybFoam.

**Bottom line:** the Python-ModelSpec approach is proven for laminar and the harness
is in place; scaling it to real closures is gated by a small set of missing NeoN
Python primitives (field algebra + math + tensor/wall-distance), not by the design.
