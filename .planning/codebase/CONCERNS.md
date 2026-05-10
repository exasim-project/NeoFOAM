# Codebase Concerns

**Analysis Date:** 2026-05-10
**Branch:** stack/distributed

---

## Technical Debt

**Duplicate mesh-reading functions across two source files:**
- Issue: `computeOffset()`, `computeNBoundaryFaces()`, and `readOpenFOAMMesh()` are defined in both `src/datastructures/foamMesh.cpp` and `src/datastructures/meshAdapter.cpp`. The `foamMesh.cpp` version is a non-distributed stub without proc-patch reordering logic; `meshAdapter.cpp` is the live version. The stub file has an unfixed `// FIXME` at its own `computeOffset` definition (line 14).
- Files: `src/datastructures/foamMesh.cpp`, `src/datastructures/meshAdapter.cpp`
- Impact: ODR risk if both translation units are linked; maintenance divergence already present (`foamMesh.cpp` lacks proc-patch separation logic that `meshAdapter.cpp` has).
- Fix approach: Delete `src/datastructures/foamMesh.cpp` or reduce it to a forwarding shim to `meshAdapter.cpp`.

**Duplicate neighbour-rank computation in `readOpenFOAMMesh`:**
- Issue: `computeNeighbRank(mesh)` is called twice in `src/datastructures/meshAdapter.cpp` lines 194–197, storing identical results in both `neighbRank` and `neighbourRank`.
- Files: `src/datastructures/meshAdapter.cpp:194-197`
- Impact: Dead assignment; wasted call. Low severity but indicative of copy-paste residue.
- Fix approach: Remove the first call and the unused `neighbRank` variable; use `neighbourRank` directly.

**`PDESolver` stores members without in-class initialisers:**
- Issue: `needReference_`, `pRefCell_`, and `pRefValue_` in `include/NeoFOAM/datastructures/pdeSolver.hpp` (lines 284–286) are declared but never initialised in the constructor member-initialiser list. `needReference_` defaults to garbage; if `setReference` is never called the subsequent `if (needReference_)` check in `solveImpl` is undefined behaviour.
- Files: `include/NeoFOAM/datastructures/pdeSolver.hpp:32-45`, `include/NeoFOAM/datastructures/pdeSolver.hpp:280-286`
- Impact: Potential UB on first solve if `setReference` is not called before `solve()`.
- Fix approach: Add `needReference_(false)`, `pRefCell_(0)`, `pRefValue_(0)` to the constructor initialiser list.

**`comparison.hpp` / `comparison.cpp` marked test-only but shipped in the production library header:**
- Issue: `include/NeoFOAM/auxiliary/comparison.hpp` carries the comment "this should be part of the tests". It is included by the generated `NeoFOAM.hpp` umbrella header (line 7) and therefore compiled into every consumer of the library.
- Files: `include/NeoFOAM/auxiliary/comparison.hpp`, `build/develop/include/NeoFOAM/NeoFOAM.hpp:7`
- Impact: Leaks OF field comparison operators into all library users; pollutes the public API.
- Fix approach: Move `comparison.hpp`/`comparison.cpp` into `test/` and remove the include from the generated umbrella header.

**`PDESolver` header contains `TODO: move to cellCentred dsl?` and unresolved design comment:**
- Issue: The `PDESolver` class in `include/NeoFOAM/datastructures/pdeSolver.hpp` (line 3) carries an open architectural question about where it belongs. The related `TODO: implement flag if matrix is assembled or not` (line 19) means callers of `computeRAU` and `computeRAUandHByA` must remember to call `assemble()` manually, with no runtime guard.
- Files: `include/NeoFOAM/datastructures/pdeSolver.hpp:3,19`, `src/algorithms/pressureVelocityCoupling.cpp:40`
- Impact: Calling `computeRAU` on an unassembled system silently reads uninitialised matrix data.
- Fix approach: Add an `isAssembled_` flag; throw or auto-assemble in `computeRAU`.

**Commented-out IC preconditioner sign-flip workaround:**
- Issue: `include/NeoFOAM/datastructures/pdeSolver.hpp:232-255` contains a large commented-out block tagged `// FIXME` and `// TODO NOTE: This is a temporary solution`. The comment states the IC preconditioner requires negating the system matrix, producing `-p` instead of `p`. The workaround is disabled but the root cause is not fixed.
- Files: `include/NeoFOAM/datastructures/pdeSolver.hpp:232-255`
- Impact: Using `preconditioner::Ic` as a standalone (non-Schwarz) preconditioner for the pressure equation on a single rank will silently converge to `-p`.
- Fix approach: Investigate sign convention in NeoN's `preconditioner::Ic` and either fix it there or re-enable the negation path with a clear explanation.

---

## Known Issues

**SurfaceField dual-storage desync → stale proc-face values:**
- Symptoms: Kernels reading `internalVector()[nIntF + bfaceii]` see a different value than `boundaryData().value()[bfaceii]` for the same face after any operation that writes only one storage location.
- Files: All kernels in `src/algorithms/pressureVelocityCoupling.cpp` that write both `iPhi[facei]` and `bvalue[bfacei]` (e.g., lines 139-141, 178-180, 248-249, 279-280), `include/NeoFOAM/auxiliary/readers.hpp:315-317`.
- Trigger: Any operation that updates `internalVector()` boundary tail without a corresponding write to `boundaryData().value()`, or vice versa. A missing `correctBoundaryConditions()` call before a proc-boundary kernel sends stale zeros to neighbours.
- Workaround: Each mutating kernel in `pressureVelocityCoupling.cpp` explicitly writes both storage slots; `constructFrom` in `readers.hpp` writes `out.boundaryData().value()` separately. But there is no compiler/runtime enforcement — future kernels can easily miss one.

**`removeBoundaryContributions` is inexact for the divergence operator (UEqn diagonal):**
- Symptoms: After stripping boundary contributions from the assembled `UEqn` matrix, the diagonal does not match OpenFOAM's `diag()` for operators other than pure Laplacian.
- Files: `test/test_distributedPressureVelocityCoupling.cpp:209-213`, `test/test_distributedPressureVelocityCoupling.cpp:344-350` (both comment out diagonal comparison with `// NOTE removeBoundaryContributions is not working in distributed case`).
- Trigger: `ddt + div(phi, U) - laplacian(nu, U)`: the div operator stores `F*w*c` in the diagonal but `F*(1-w)*c` in `nonLocal`, so the restoration residual is `F*(w - (1-w))` which is non-zero for `w != 0.5`.
- Workaround: Diagonal validation is done indirectly through the `rAU` section.

**`readSurfaceBoundaryConditions` does not handle `procBoundary0to1` patch names:**
- Symptoms: Reading a decomposed surface field with a `processorCyclicFvPatch` named `procBoundary0to1` (or similar) falls through the `patchInserter` map lookup and produces an out-of-range `std::map::operator[]` access (undefined behaviour / exception).
- Files: `include/NeoFOAM/auxiliary/readers.hpp:174` (`// TODO this approach fails for procBoundary0to1`)
- Trigger: Cases using `cyclic` + domain decomposition, which produce `processorCyclic` patches on decomposed boundaries.
- Workaround: None currently; use simple decomposition without cyclic patches.

**`fixedValue` parallel BC parsing falls back to `empty` when token count is 1:**
- Symptoms: In parallel, `ofVolField.boundaryField().writeEntries()` can serialise a `fixedValue` boundary condition with only one token (the uniform value), causing `tokenList.size() == 1`. The code then tags the patch as `"empty"`, silently dropping the boundary condition.
- Files: `include/NeoFOAM/auxiliary/readers.hpp:67-109` (`// NOTE FIXME in parallel cases we end up with token.size ==1`)
- Trigger: Any parallel run with a `fixedValue` patch whose serialised form has only a single token (e.g. scalar zero `0`).
- Workaround: Unknown; affects correctness of BCs silently.

**`SetReference` is rank-0 only — no global cell index mapping:**
- Symptoms: Calling `pEqn.setReference(globalCellIdx, pRefValue)` on any rank pins `globalCellIdx` as a local index on rank 0 only. If the physical reference cell does not live on rank 0, the pressure level is not fixed and the solver produces an underdetermined system.
- Files: `include/NeoFOAM/datastructures/pdeSolver.hpp:83-100` (`// FIXME: assume rank 0 owns the reference`), `include/NeoFOAM/datastructures/pdeSolver.hpp:136-143`.
- Trigger: Any mesh partition where the reference cell (cell 0 by default in `neoIcoFoam`) is not on rank 0. Can happen with `scotch`, `metis`, or any non-trivial decomposition.
- Workaround: Current usage always passes local index 0 from rank 0, and `neoIcoFoam.cpp:134-137` gates the call with `if (ofP.needReference() && pRefCell >= 0)`. Safe only for uniform decompositions starting at global cell 0.

---

## Architectural Risks

**Tests hard-coded to exactly 3 MPI processes:**
- All three distributed test binaries (`test_distributedUnstructuredMesh`, `test_distributedMomentum`, `test_distributedPressureVelocityCoupling`) contain `REQUIRE(Foam::Pstream::nProcs() == 3)`.
- Files: `test/test_distributedUnstructuredMesh.cpp:26`, `test/test_distributedMomentum.cpp:25`, `test/test_distributedPressureVelocityCoupling.cpp:25`
- Risk: No coverage of 2-rank, 4-rank, or 8-rank decompositions. Bugs that only appear with non-linear-chain topology (scotch, metis, hierarchical) are not caught. The weight-symmetry and displacement-sorting tests acknowledge this explicitly.

**Proc-face weight-reversal regression not detectable with uniform mesh:**
- The `interpolate rAU` section in `test_distributedPressureVelocityCoupling.cpp:109-160` uses `simpleGrading (1 1 1)`, giving all proc faces `w == 0.5`. Both `w*own + (1-w)*ghost` and `(1-w)*own + w*ghost` produce identical results when `w == 0.5`, so a formula inversion in the proc-face branch of `computeLinearInterpolation` would pass all current tests.
- Files: `test/test_distributedPressureVelocityCoupling.cpp:125-133`
- Risk: Weight reversal would cause solution divergence on any non-uniform mesh in production (e.g. `tutorials/cylinder2D`).

**`procFaceCheck.hpp` uses raw `MPI_COMM_WORLD` — not portable to sub-communicator runs:**
- The diagnostic utility `checkProcFaceConsistency` in `include/NeoFOAM/auxiliary/procFaceCheck.hpp:221,250,261,300` calls `MPI_Comm_rank(MPI_COMM_WORLD, ...)` and performs all exchanges on `MPI_COMM_WORLD` rather than the communicator used by OpenFOAM's `Pstream`.
- Files: `include/NeoFOAM/auxiliary/procFaceCheck.hpp:221-300`
- Risk: Will silently exchange data with wrong ranks if `FOAM_COMM` or a custom communicator is active (e.g. in coupled solver contexts). Currently only a diagnostic utility, but the pattern could be copied into production code.

**Proc-patch weight-symmetry sort is a no-op on the current test mesh:**
- The comment in `test_distributedUnstructuredMesh.cpp:64-76` notes that `collectProcPatchOffsets` sorts offsets by neighbour rank, but the existing 3-rank `simpleDecomposition` mesh already has patches in ascending rank order, so the sort is never exercised. A scotch-decomposed 4+ rank mesh would test the actual sort path.
- Files: `test/test_distributedUnstructuredMesh.cpp:64-76`
- Risk: The sort fix in `basicGeometryScheme.cpp` (NeoN submodule) is unverified for the cases it was actually introduced to fix.

**`foamMesh.cpp` non-distributed mesh reader is a dead code path:**
- `src/datastructures/foamMesh.cpp` implements a serial `readOpenFOAMMesh` (no proc-patch reordering, no `nProcPatches`, no `neighbourRank`). It is never called from `meshAdapter.cpp`; all constructor paths call the overloaded version in `meshAdapter.cpp`. The file has no `CMakeLists.txt` entry visible in the src tree.
- Files: `src/datastructures/foamMesh.cpp`
- Risk: If mistakenly linked it shadows the correct implementation; if silently excluded it is wasted maintenance burden.

**`fullMeshOnGPU` defaults to `false`; mesh geometry stays on CPU:**
- `readOpenFOAMMesh` in `src/datastructures/meshAdapter.cpp:147` takes `bool fullMeshOnGPU = false`, placing `mesh.points()` on `SerialExecutor` while other arrays go to the target executor. This is an intentional optimisation, but it silently creates a mixed-executor mesh that can cause assertion failures if any NeoN kernel assumes all mesh arrays share the same executor.
- Files: `src/datastructures/meshAdapter.cpp:156`, `include/NeoFOAM/datastructures/meshAdapter.hpp:27-30`
- Risk: Future GPU kernels that access `mesh.points()` will trigger an out-of-device-memory access or an executor mismatch assertion.

---

## Open Questions

**Where does `PDESolver` belong?**
- `include/NeoFOAM/datastructures/pdeSolver.hpp:3` asks whether `PDESolver` should move into NeoN's `cellCentred` DSL layer. The answer determines whether the OpenFOAM `RunTime` dependency should be removed from the solver interface, which would make the class more reusable and testable in isolation.

**Global pressure reference cell convention for arbitrary partitions:**
- The current convention (rank 0, local index 0) is embedded in `pdeSolver.hpp` and `neoIcoFoam.cpp`. There is no utility to convert a global cell index to `(rank, localIdx)`. When the mesh is partitioned by scotch or metis, physical cell 0 can land on any rank.
- File: `include/NeoFOAM/datastructures/pdeSolver.hpp:83-100`

**`noSlip` boundary condition for vector fields:**
- `include/NeoFOAM/auxiliary/readers.hpp:112` (`// TODO specialize for vector`) notes that `noSlip` is handled generically via `fixedValue` with zero-initialised `type_primitive_t {}`. This works for vectors but the intent is to add a proper specialisation. It is unclear whether the generic path is correct for all primitive types.

**`adjustPhi` and `constrainPressure` missing from PISO loop:**
- `examples/neoIcoFoam/neoIcoFoam.cpp:119-122` disables both `Foam::adjustPhi(phiHbyA, U, p)` and `Foam::constrainPressure(p, U, phiHbyA, rAU)`. These calls enforce flux consistency at fixed-flux and fixed-value pressure boundaries. Without them, mass imbalance can accumulate across PISO iterations on cases with mixed boundary types.
- File: `examples/neoIcoFoam/neoIcoFoam.cpp:119-122`

**`continuityErrs.H` missing from PISO loop:**
- `examples/neoIcoFoam/neoIcoFoam.cpp:147-148` omits continuity error reporting. This makes it impossible to monitor divergence of the continuity equation during a run, which is the primary indicator of pressure-solver failure.

---

## Priority Items

1. **[High] Uninitialised `needReference_` in `PDESolver`** — UB on every solve where `setReference` is not called. Fix: add default initialisers to constructor.
   - `include/NeoFOAM/datastructures/pdeSolver.hpp:32-45,284-286`

2. **[High] `fixedValue` BC silently replaced by `empty` in parallel** — Incorrect boundary conditions applied without any error. Fix: diagnose the token-count-1 path and parse the value correctly.
   - `include/NeoFOAM/auxiliary/readers.hpp:67-109`

3. **[High] `procBoundary0to1` name not handled in `readSurfaceBoundaryConditions`** — Crashes or silent UB on cyclic+parallel cases. Fix: add `processorCyclic` to the `patchInserter` map.
   - `include/NeoFOAM/auxiliary/readers.hpp:174`

4. **[Medium] Proc-face weight-reversal test gap** — The interpolation test cannot catch reversed-weight bugs. Fix: add a graded-mesh (`simpleGrading != 1`) setup to `setup_pressureVelocityCoupling`.
   - `test/test_distributedPressureVelocityCoupling.cpp:125-133`

5. **[Medium] Reference cell rank-0 assumption** — Will fail silently with scotch/metis decomposition. Fix: implement a global→(rank, localIdx) cell lookup and thread it into `setReference`.
   - `include/NeoFOAM/datastructures/pdeSolver.hpp:83-100`

6. **[Medium] Duplicate functions in `foamMesh.cpp` / `meshAdapter.cpp`** — ODR risk and maintenance divergence. Fix: delete `foamMesh.cpp` or ensure it is not compiled.
   - `src/datastructures/foamMesh.cpp`, `src/datastructures/meshAdapter.cpp`

7. **[Medium] `comparison.hpp` in production library umbrella header** — Pollutes public API with test-only operators. Fix: move to `test/`.
   - `build/develop/include/NeoFOAM/NeoFOAM.hpp:7`

8. **[Medium] `adjustPhi` and `constrainPressure` missing from `neoIcoFoam`** — Mass imbalance on mixed-BC cases.
   - `examples/neoIcoFoam/neoIcoFoam.cpp:119-122`

9. **[Low] Duplicate `computeNeighbRank` call in `readOpenFOAMMesh`** — Dead assignment, minor cleanup.
   - `src/datastructures/meshAdapter.cpp:194-197`

10. **[Low] `procFaceCheck.hpp` hardcodes `MPI_COMM_WORLD`** — Not a production issue today but a copy-paste hazard.
    - `include/NeoFOAM/auxiliary/procFaceCheck.hpp:221,250,261,300`

---

*Concerns audit: 2026-05-10*
