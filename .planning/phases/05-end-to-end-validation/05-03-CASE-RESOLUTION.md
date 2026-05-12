---
phase: 05-end-to-end-validation
plan: 03
type: resolution
status: complete
---

## Root Cause Analysis

**First diverging output time:** t=0.005 (only output time; L∞_U=4.77, L∞_p=83.5)

**First procFaceCheck failure:** `[procFaceCheck] U after momentumPredictor solve: FAILED`
— triggered on timestep 1, rank 0 and 1, patch 0 (rank 0↔1 boundary), face 326.

---

### What Fails

After `UEqn.solve(-1.0 * dsl::exp::grad(p))` in the momentum predictor
([examples/neoIcoFoam/neoIcoFoam.cpp:99](examples/neoIcoFoam/neoIcoFoam.cpp)),
the NeoN linear solver updates the interior cell values of `U` but does **not** automatically
update the processor boundary face data. Without an explicit `U.correctBoundaryConditions()` call,
every rank's processor ghost cells remain at their previous value — the initial condition
`(1.0, 0.0, 0.0)` m/s — while the interior of the neighboring rank has already solved to
physically different values.

The procFaceCheck log confirms this:
```
local.boundary  = (1.000000e+00, 0.000000e+00, 0.000000e+00)   ← IC / stale
remote.internal = (5.736177e+00, 0.000000e+00, 0.000000e+00)   ← solved interior
```

These extreme values on the first timestep (5.7 m/s from IC 1 m/s) are caused by the stale
processor boundary feeding wrong neighbour values into the diffusion/convection stencil, driving
the Ginkgo linear system to an inconsistent solution that compounds every iteration.

`U after rotateOldTimes: OK` confirms the field starts consistent; the corruption happens
exclusively inside `UEqn.solve()` — specifically during the post-solve phase where NeoN
does not automatically push new interior values to processor ghost cells.

---

### Why This Happens

In OpenFOAM's `fvMatrix<T>::solve()`, the underlying `lduMatrix` solve path calls
`psi.correctBoundaryConditions()` on the solution field after convergence. NeoN's
`PDESolver<T>::solve()` does **not** have an equivalent automatic call — `correctBoundaryConditions()`
must be called explicitly by the caller.

The post-PISO `updateVelocity` step (line 229–230) already does this correctly:
```cpp
nf::updateVelocity(hByA, crAU, p, U);
U.correctBoundaryConditions();   // ← present and correct
```

The momentum predictor path (line 95–101) is missing the same call:
```cpp
if (piso.momentumPredictor())
{
    UEqn.solve(-1.0 * dsl::exp::grad(p));
    // ← U.correctBoundaryConditions() MISSING HERE
    nf::checkProcFaceConsistency(U, "U after momentumPredictor solve");
}
```

---

### Proposed Fix for Plan 05-04

**File:** `examples/neoIcoFoam/neoIcoFoam.cpp`  
**Location:** after line 99 (`UEqn.solve(...)`) inside the `if (piso.momentumPredictor())` block

```cpp
if (piso.momentumPredictor())
{
    UEqn.solve(-1.0 * dsl::exp::grad(p));
    U.correctBoundaryConditions();   // ← ADD THIS LINE
    nf::checkProcFaceConsistency(U, "U after momentumPredictor solve");
}
```

**Why this is sufficient:** `U.correctBoundaryConditions()` performs the MPI_Sendrecv exchange
for all processor patches, pushing each rank's new interior values into the neighboring rank's
ghost cells. After this call, `checkProcFaceConsistency` should report OK.

The PISO loop downstream (`computeRAUandHByA`, `constrainHbyA`, `flux(hByA)`, `ddtFluxCorr`)
all read U boundary values when computing face fluxes. With correct ghost cells, the HbyA
interpolation and the pressure Poisson system will be consistent across ranks.

---

### Confidence Assessment

- **High confidence** this is the primary bug: the first procFaceCheck failure is exactly at the
  momentum predictor solve on timestep 1, all 4 ranks report FAILED, and the stale value is
  precisely the IC value `(1, 0, 0)`.
- **Medium confidence** this is the only bug needed for correctness: subsequent checks (p, phi,
  U after updateVelocity) are not shown in the log. Plan 05-04 should run with the fix applied
  and check whether the remaining procFaceCheck points all report OK.
- If additional FAILEDs appear after the fix, they indicate secondary bugs (e.g., in
  `p.correctBoundaryConditions()` or `updateFaceVelocity`) that were masked by the primary failure.
