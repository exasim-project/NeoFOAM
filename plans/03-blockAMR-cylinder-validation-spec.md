<!--
SPDX-License-Identifier: GPL-3.0-or-later
SPDX-FileCopyrightText: 2026 NeoFOAM authors
-->

# Spec 03 — Cylinder case, observables & single-solver validation

**Status:** Partially implemented (2026-07-09) · **Depends on:** Specs 01 (solver) + 02 (code-verified) · **Blocks:** Spec 04

## Implementation status (2026-07-09)

**Done & passing (fast suite):**
- **P1 / INT-7** — Ghia lid-cavity through the framework (`test_cavity_ghia.py`). Cavity case
  `cases/cavity/` is a 16×16 x-y square with a thin **nz=4** periodic z-slab (extent 0.25 →
  dz=1/16=dx cubic; 4× cheaper than the 16³ cube, correct for a homogeneous-z 2D benchmark).
- **P2 / P3 / INT-8** — `postpro.py` observables (`force_coefficients` via the recorded
  direct-forcing IBM reaction force, momentum-deficit CV fallback; `strouhal` FFT;
  `recirculation_length`; `separation_angle`) + 9 canned unit tests (`test_postpro.py`).
- **P7** — config-driven cylinder case (Cartesian domain + `eb` cylinder + inlet/outlet/slip
  walls), per-Re via `fvSolution` `nu`. Smoke E2E `test_cylinder_run.py` (10 steps).

**The high-res cylinder fix (was the open blocker): a CFL constraint, not a solver bug.**
At CFL 0.3 the 160×80 outflow+IBM run diverges/crashes; at **CFL 0.1** it is stable
(max|U|≈1.43 bounded, MLMG → ~1e-7). True with *both* the engine's default `smoother`
bottom and `bicgstab` → purely the timestep, **no engine change / no env toggles**. (Also:
use **nz≥8** so the nodal `MLNodeLaplacian` multigrid can coarsen; nz=4 → "MLMG failed".)

**P4/P5/P6 (literature bands) — NOT met; blocked on cut-cell EB.** A converged Re=20 run at
the stable 160×80 geometry gives **Cd ≈ 3.3**, vs the unconfined literature ≈2.05 — inflated
by (a) the **direct-forcing staircased** body (first-order surface, biases Cd high) and (b)
**20 % domain blockage** (D=0.2 in LY=1; low blockage needs a big domain → the nz-coarsening
limit + GPU-OOM/hours). The tight bands need the deferred **cut-cell EB** (exact surface
forces) + a low-blockage domain. Encoded as an **opt-in** validation test
(`test_cylinder_validation.py`, gated on `NEOFOAM_BLOCKAMR_VALIDATION=1`) that asserts the
*achievable* band + qualitative wake, with the modelling margin stated. Re=40/Re=100 (E2E-2/3)
likewise deferred to the EB iteration.

---

**Parent roadmap:** [`incompressibleFluidBlockAMR-verification.md`](./incompressibleFluidBlockAMR-verification.md) §2.2–2.4, §3.2
(covers roadmap Phases 4–5; requirements **R4, R5, R6, R7**; tests **INT-7, INT-8, E2E-1/2/3**)

## Scope

Give the solver a **flow-around-a-cylinder** case and the post-processing to
extract mesh-independent **observables** (`Cd`, `Cl'`, `St`, `Lr/D`, `θs`), then
validate the new solver *on its own* against literature bands:
- **INT-7** Ghia lid-cavity (self-consistency benchmark through the framework solver),
- **INT-8** force post-processing unit test on a canned field,
- **E2E-1/2** steady cylinder Re=20 & 40 (MUST),
- **E2E-3** unsteady cylinder Re=100 (SHOULD).

Cross-solver agreement and performance are **Spec 04**. This spec makes the new
solver *individually trustworthy* on the target problem.

## Motivation

Spec 02 proves the math; this spec proves the solver reproduces a real,
literature-documented external flow. It also builds the `Cd/Cl/St/Lr` extraction
that Spec 04's cross-solver comparison consumes — so the observable definitions
live here, once.

## Reference observables (laminar circular cylinder)

| Observable | Re=20 | Re=40 | Re=100 |
|---|---|---|---|
| `Cd` | ≈2.05 | ≈1.54 | ≈1.32–1.36 (mean) |
| `Lr/D` (recirc.) | ≈0.93 | ≈2.2–2.3 | — |
| `θs` (separation) | ≈43–45° | ≈53° | — |
| `Cl'` (RMS/amp) | 0 | 0 | ≈0.30–0.35 |
| `St = fD/U` | — | — | ≈0.164–0.172 |

Re set via `Re = U∞·D/ν`. Derive geometry/`D` from `tutorials/cylinder2D`
(`D=0.02 m`) so Spec 04's other solvers model the same physical setup. The
**Schäfer–Turek (1996) 2D-2** confined benchmark is the recommended Re=100 target.

## Requirements

| ID | Requirement | Priority | Verify |
|---|---|---|---|
| P1 | Ghia lid-cavity centreline through the framework solver matches Ghia (1982) within literature tol | MUST | INT-7 |
| P2 | `postpro.py` computes `Cd`/`Cl` from a solution field with correct sign/order on a canned input | MUST | INT-8 |
| P3 | `St` via FFT of `Cl(t)`; `Lr/D` via wake-centreline `u=0` crossing; `θs` via surface separation | MUST | INT-8 + E2E-3 |
| P4 | blockAMR cylinder Re=20: `Cd`, `Lr/D`, `θs` inside literature bands | MUST | E2E-1 |
| P5 | blockAMR cylinder Re=40: `Cd`, `Lr/D`, `θs` inside literature bands | MUST | E2E-2 |
| P6 | blockAMR cylinder Re=100: shedding onset; time-mean `Cd`, `Cl'`, `St` inside bands | SHOULD | E2E-3 |
| P7 | The cylinder case is config-driven (Cartesian domain + EB cylinder + inlet/outlet/slip walls), per-Re parameterised | MUST | E2E-1 setup |

## Architecture

```python
# postpro.py  (in the solver package — the single source of observable defs)
def force_coefficients(engine, U_inf, D, rho, nu) -> tuple[float, float]   # (Cd, Cl)
    # EB surface integral of pressure + viscous traction if EB face data is
    # exposed by src/NeoN/src/bindings/blockAMR/; else momentum-deficit
    # (control-volume) fallback over a wake box.
def strouhal(cl_series: np.ndarray, dt: float, D: float, U_inf: float) -> float   # FFT peak
def recirculation_length(U_field, geom, D) -> float     # centreline u=0 crossing behind cylinder
def separation_angle(U_field, geom, cylinder) -> float
```

- **Cylinder case config** extends Spec 01's `MeshDictConfig` with
  `eb.type=cylinder` (`center`, `radius=D/2`, `axis=z`) and a non-periodic domain:
  `inlet` fixedValue `(U∞,0,0)`, `outlet` zeroGradient/Neumann, top/bottom `slipWall`,
  cylinder `noSlip`. Per-Re: set `ν` (or `U∞`) for the target `Re`.
- **Ghia (INT-7)** reuses the blockamr lid-cavity check (`src/NeoN/test/blockamr/test_dsl_lid_cavity.py`
  semantics: interpolate centreline U to Ghia points, assert error < tol) but runs
  through the framework `run(["incompressibleFluidBlockAMR"])` path with a cavity config.
- **Steady detection (E2E-1/2):** stop when `‖Uⁿ⁺¹−Uⁿ‖` < tol (add a steady-state
  `loopCondition` contribution, mirroring the adaptive-time-step interface pattern),
  or run a fixed long horizon and assert the observable on the converged tail.
- **Unsteady (E2E-3):** run past shedding onset, record `Cl(t)` on the cylinder,
  time-average `Cd` over ≥5 shedding periods, FFT for `St`.

## Tasks (TDD)

1. `postpro.py`: force coefficients first (EB-surface if available, else
   momentum-deficit), then `strouhal`/`recirculation_length`/`separation_angle`.
2. **INT-8/P2,P3**: canned-field unit tests — a uniform-flow + known-perturbation
   field yields expected `Cd` sign/order; a synthetic sinusoidal `Cl(t)` yields the
   injected frequency's `St`.
3. **INT-7/P1**: cavity config + Ghia assertion through the framework solver.
4. Cylinder case builder (config + EB cylinder + BCs), per-Re. **P7.**
5. **E2E-1/E2E-2/P4,P5**: Re=20/40 steady; tune resolution/AMR until bands pass;
   assert `Cd`, `Lr/D`, `θs`.
6. **E2E-3/P6**: Re=100 unsteady; assert `St`, `Cl'`, mean `Cd` (SHOULD).

## Test logic

- **INT-8** is deterministic and fast (no solve) — it locks the observable math so
  E2E band failures point at physics, not post-processing bugs.
- **E2E-1/2** assert a *band* (`lo ≤ Cd ≤ hi`), not a point — discretisation +
  EB introduce a few-percent bias; the band is the literature spread widened by a
  stated modelling margin.
- **E2E-3** asserts `St` (robust, frequency-based) as the primary unsteady metric;
  `Cd`/`Cl'` as secondary. Marked SHOULD due to cost/variance.

## Test selector

```bash
pytest test/solver/incompressibleFluidBlockAMR/test_cylinder_*.py -q          # E2E-1/2/3
pytest test/solver/incompressibleFluidBlockAMR/ -k "ghia or force" -q          # INT-7/8
```

## Out of scope / open questions

- The other two solvers' cylinder runs and cross-solver agreement → **Spec 04**
  (this spec only validates blockAMR against literature).
- **EB surface-force availability** is the key unknown: does
  `src/NeoN/src/bindings/blockAMR/` expose EB face areas/centroids/normals? If not,
  `force_coefficients` uses the momentum-deficit method (decide in task 1).
- Confined (Schäfer–Turek) vs open-domain cylinder — pick one geometry convention
  and apply it to all solvers in Spec 04.
- Steady-state detection vs fixed-horizon for E2E-1/2 (either satisfies P4/P5).
