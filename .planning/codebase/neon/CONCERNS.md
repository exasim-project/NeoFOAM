# NeoN Concerns

**Analysis Date:** 2026-05-10
**Branch:** stack/distributed (rebased on fix/testsRebase)

---

## Active WIP Areas

The last 10 commits are all WIP or fixes for distributed correctness. Nothing on this branch is considered stable.

**Divergence operator boundary assembly** (`src/finiteVolume/cellCentred/operators/gaussGreenDiv.cpp`):
- Commit "wip div changes" (7e1a53c) is the HEAD — the proc-boundary implicit divergence coefficients are actively being revised.
- `computeDivProcBoundImpl` now uses a single local-convention formula (lines 273–280); the correctness of the sign and weight for the non-owner side has not been validated against a serial reference on a non-trivial mesh.
- The explicit `computeDivExp` path still contains `// TODO: currently we just copy the boundary values over` at line 167–168, meaning boundary-interpolated face values are approximated, not computed correctly.

**Proc-face sort order** (`src/mesh/unstructured/unstructuredMesh.cpp`):
- Commit "wip fix sort order" (50bf325) — the `computeCommunicationPattern` displacement logic was recently rewritten. The concern (documented in long comments at lines 448–466) is that the `recvIdx` must stay in mesh-boundary order; any decomposition where `neighbourRanks` is non-ascending previously silently sent data to wrong ranks.

**Processor discontinuities / weight calculation** (`src/finiteVolume/cellCentred/stencil/basicGeometryScheme.cpp`):
- Commit "wip fix processor discontinuities, reversed weight calculation in linear" (6803c2f) — the `updateWeights` kernel for proc faces (lines 211–234) now exchanges `d_own` via MPI and computes `w = d_nei / (d_own + d_nei)`. This logic is new and untested on graded or non-uniform decompositions.

**GPU fence coverage** (`include/NeoN/fields/boundaryData.hpp`, lines 265–268):
- Commit "wip add GPU fences" (4b5d473) — a `fence(boundaryData.exec())` was added before MPI sends. It is unknown whether all kernel dispatch paths through `correctBoundaryConditions` reach this fence before MPI reads device memory.

---

## Technical Debt

**`partitioning.hpp` — hard-coded 3-rank assumption**
- File: `include/NeoN/distributed/partitioning.hpp`, lines 68–112
- `partitionSurfaceField` has `FIXME this only works for 3 ranks`. The rank-0/1/2 branching is hand-written with literal index arithmetic. Deploying on anything other than 3 ranks will silently produce wrong face-index mappings.

**`computeBoundaryMatrixMapVector` is entirely commented out**
- File: `src/mesh/unstructured/boundaryMesh.cpp`, lines 130–147
- Function body is all commented-out code returning an empty vector. Marked `FIXME`. Not clear if it is used anywhere at runtime, but it is declared in the public header (`include/NeoN/mesh/unstructured/boundaryMesh.hpp` line 342: `// FIXME is this actually used`).

**Sparsity-pattern construction is entirely serial (CPU) even on GPU meshes**
- File: `src/linearAlgebra/faceToMatrixAddress.cpp`, line 183
- `setSparsityPatternFaceToMatrixAddressSerial` is called unconditionally. All offset/index arrays are copied to host, processed with a serial `parallelFor(SerialExecutor{}, ...)`, then copied back to device. This is an O(nFaces) CPU bottleneck on every time step if the linear system is re-created, and a correctness hazard if the mesh is GPU-resident and the copy is ever skipped.

**Communication pattern is recomputed on every `correctBoundaryConditions` call**
- Files: `src/finiteVolume/cellCentred/fields/volumeField.cpp` line 127, `src/finiteVolume/cellCentred/fields/surfaceField.cpp` line 49
- Both call `computeCommunicationPattern(this->mesh())` on every boundary correction. This triggers `MPI_Alltoallv` for the global-offset exchange and several `std::vector` allocations per call. On a time-stepping loop this is called multiple times per iteration.
- Comment: `FIXME dont recompute communication pattern`

**`diagonalSolver.hpp` — `solveDist` stubs are no-ops**
- File: `include/NeoN/linearAlgebra/diagonalSolver.hpp`, lines 28–38
- Both `solveDist(scalar)` and `solveDist(Vec3)` return empty `SolverStats {}` and do nothing. Calling `DiagonalSolver` in distributed mode silently produces a zero solution vector with no error.

**`petsc.hpp` — `clone()` aborts, residual norms missing**
- File: `include/NeoN/linearAlgebra/petsc.hpp`, lines 73–76 and 126
- `clone()` calls `NF_ERROR_EXIT("Not implemented")` — storing a `PetscSolver` in any container that clones solvers will abort at runtime.
- Residual norms are not extracted; the returned `SolverStats` always has `{numIter, 0.0, 0.0, 0.0}`.

**`BoundaryMesh::nBoundaries()` is marked FIXME**
- File: `include/NeoN/mesh/unstructured/boundaryMesh.hpp`, line 244
- `localIdx nBoundaries() const { return offset_.size() - 1; }` is marked `// FIXME`. The semantics of `offset_` are entangled with whether proc patches are included, making patch-count arithmetic error-prone across callers.

**Inconsistent return types in `BoundaryMesh` accessors**
- File: `include/NeoN/mesh/unstructured/boundaryMesh.hpp`, lines 82 and 236
- Both `faceCells()` (returns `labelVector`) and `offset()` (returns `std::vector<localIdx>`) are noted with `TODO either dont mix return types` / `TODO consistent use of Vector on CPU`. The mix forces host-side logic to call `.copyToHost()` for some and not others.

**`unstructuredMesh.cpp` create1DUniformMeshPart has dead GPU path**
- File: `src/mesh/unstructured/unstructuredMesh.cpp`, lines 413–418
- Comment: `// FIXME doesnt work on GPU`. The GPU code path for setting face-normal signs on non-zero ranks is commented out and replaced with a host-copy workaround that swaps entries on rank 2. All proc-boundary sign correction is therefore CPU-only and done at mesh construction time rather than in a portable kernel.

**`scaledInvDiagNegLUx` — O(nProcFaces²) inner loop**
- File: `include/NeoN/linearAlgebra/linearSystem.hpp`, lines 439–453
- The non-local (proc-boundary) off-diagonal contribution is accumulated by scanning `nonLocalRows` linearly for every cell row. Comment: `// FIXME this scans everytime all boundary values`. For a mesh with many proc-boundary faces this is O(nCells × nProcFaces) per PISO iteration.

**`removeBoundaryContributions` — missing RHS subtraction for non-local matrix**
- File: `include/NeoN/linearAlgebra/linearSystem.hpp`, line 376–377
- The non-local matrix loop does `Kokkos::atomic_add` to the diagonal but the matching `Kokkos::atomic_sub(&rhs[celli], bRhs[facei])` is commented out with `// FIXME add`. This makes `removeBoundaryContributions` (used in testing) asymmetric between local and non-local boundary contributions.

**`deltaCoeffs` for proc faces uses only local owner-to-face distance**
- File: `src/finiteVolume/cellCentred/stencil/basicGeometryScheme.cpp`, lines 276–295
- Comment: `FIXME proc-face deltaCoeffs should mirror updateNonOrthDeltaCoeffs and use the cell-to-cell distance across the cut (d_own + d_nei) instead of the local owner-to-face distance`. The current code produces an asymmetric Laplacian coefficient at proc boundaries on any non-uniform mesh.

**`geometryScheme.cpp` — no selection mechanism for geometry scheme**
- File: `src/finiteVolume/cellCentred/stencil/geometryScheme.cpp`, line 79
- `kernel_(std::make_unique<BasicGeometryScheme>(mesh))` is hard-coded. The comment `// TODO add selection mechanism` indicates planned extensibility that is not yet implemented.

**`dsl/explicit.hpp` and `dsl/implicit.hpp` are coupled to fvcc**
- Files: `include/NeoN/dsl/explicit.hpp` line 13, `include/NeoN/dsl/implicit.hpp` line 12
- Both marked `// TODO: decouple from fvcc`. The DSL layer depends on finite-volume cell-centred types directly, blocking reuse for other discretisation approaches.

---

## Known Correctness Issues

**`faceToMatrixAddress.hpp` diagOffset/ownerOffset/neighbourOffset are incorrect in distributed mode**
- File: `include/NeoN/linearAlgebra/faceToMatrixAddress.hpp`, line 127
- Comment: `// FIXME these are probably incorrect in distributed mode`
- These offsets index within a row of the CSR matrix. In distributed mode the row for a proc-boundary cell may include a non-local column entry that is not counted during `setSparsityPatternFaceToMatrixAddressSerial` (which only iterates internal faces). Any code calling `diagIdx()`, `upperIdx()`, or `lowerIdx()` for a proc-adjacent cell may compute the wrong flat index into the values array.

**`setProcBoundarySparsityPattern` — missing MPI communication for col indices**
- File: `src/linearAlgebra/faceToMatrixAddress.cpp`, line 169
- Comment: `// FIXME needs communication to other side`
- The proc-boundary column index (`colIdx`) for the ghost cell is received via `computeCommunicationPattern` (`recvIdx`). This communication is done in `createSparsityPatternFaceToMatrixAddress` (line 314), but `setProcBoundarySparsityPattern` itself leaves the column-index determination incomplete, relying on the `recvIdx` vector being pre-populated. If `recvIdx` is stale or empty (e.g. on a non-distributed mesh fallback), the proc-boundary off-diagonal entries have wrong column indices.

**`processor.hpp` — boundary exchange not implemented**
- File: `include/NeoN/finiteVolume/cellCentred/boundary/volume/processor.hpp`, lines 19–20
- Comment: `// FIXME TODO exchange values on boundaries with neighbour rank`
- `setProcBoundaryValue` copies the local owner-cell value into the boundary data but does NOT communicate with the neighbour rank. The boundary data slot for the ghost cell is therefore the local cell value (zero-gradient extrapolation), not the actual neighbour value. The actual MPI exchange is deferred to `VolumeField::correctBoundaryConditions` (in `volumeField.cpp`), but only for `boundaryData().value()`. Any operator that reads `boundaryData().refValue()` or other slots will see un-exchanged data.

**`SurfaceField` internalVector/boundaryData sync gap**
- Context from memory: Both `internalVector()[nIntF..]` and `boundaryData().value()` must stay in sync or MPI sync sends stale zeros.
- `SurfaceField::correctBoundaryConditions` in `src/finiteVolume/cellCentred/fields/surfaceField.cpp` exchanges only `boundaryData().value()`. The proc-face entries of `internalVector()` (indices `[nIntF+nBnd, nTotal)`) are set by `setProcBoundaryValue` (zero-gradient from local owner), but are NOT overwritten with the received ghost value. Any kernel that reads `internalVector()[facei]` for a proc face will see the local extrapolation, not the communicated neighbour value.

**`std::abs` portability on GPU**
- File: `src/finiteVolume/cellCentred/stencil/basicGeometryScheme.cpp`, line 215
- Comment: `// FIXME is std::abs available on GPU?`
- The `updateWeights` proc-face kernel uses `std::abs(static_cast<scalar>(...))` inside a `NEON_LAMBDA`. On CUDA/HIP device code, `std::abs` may not be available; the correct GPU-portable call is `Kokkos::abs` or a device-annotated helper. This will cause a compilation error or silent wrong result on GPU builds.

**`ginkgoDistributed.cpp` — Ginkgo matrix and solver are recreated on every solve**
- File: `src/linearAlgebra/ginkgo/ginkgoDistributed.cpp`, lines 105, 236
- Comments: `// TODO dont recreate`, `// TODO dont re-init`
- The `gko::experimental::distributed::Matrix` and the solver factory generator are re-created each call to `solveDist`. This includes the `build_partition_from_local_size` and full matrix assembly inside Ginkgo, which are expensive and trigger device-host transfers. On a PISO loop this runs every pressure solve.

---

## Interface Instability

**`LinearSystem` non-local matrix API is evolving**
- File: `include/NeoN/linearAlgebra/linearSystem.hpp`
- The `nonLocalMatrix_` member, the `CommunicationPattern` embed, and the `foldNonLocalIntoLocal()` method are recent additions. Their signatures are changing with every WIP commit. NeoFOAM code that calls `ls.nonLocalMatrix()` or `ls.foldNonLocalIntoLocal()` should be insulated behind a thin wrapper.

**`BoundaryMesh` offset/patch-count API**
- `nBoundaries()`, `nBoundaryFaces()`, `nProcBoundaryFaces()`, `nProcBoundaryPatches()` all have open FIXMEs or inconsistent semantics. Callers should use only `nBoundaryFaces()` and `nProcBoundaryFaces()` and avoid `nBoundaries()` until the FIXME is resolved.

**`BasicGeometryScheme::updateWeights` proc-face section**
- The MPI exchange path (`exchangeProcOwnerDistance`) was added in the most recent rounds of WIP commits. Its return type, call site, and the `dExchange` variable name suggest it may be renamed or refactored once the proc-face weight formula is validated.

**`computeDivProcBoundImpl` sign/weight convention**
- The comment block in `gaussGreenDiv.cpp` lines 241–283 documents a recently rationalised formula that replaced an `isOwner` flag. This is the current HEAD of WIP. The formula itself is plausible but the weight `bweights[facei]` (line 274) indexes `weights.internalVector()` using the full face index — this must be verified to match the face-index convention used by `updateWeights`.

---

## Distributed / MPI Risks

**Face-index convention: compressed vs. full**
- `mesh.faceCentres()` / `mesh.faceOwner()` / `mesh.faceNeighbour()` / `mesh.faceAreas()` are sized `nTotalFaces` (internal + all boundary patches, in OpenFOAM-full order).
- `mesh.boundaryMesh().cf()` / `.sf()` / `.faceCells()` / `.deltaCoeffs()` are compressed, sized `nBoundaryFaces + nProcBoundaryFaces`.
- Kernels iterating `[nIntF + nBnd, nTotalFaces)` that read `mesh.faceCentres()[facei]` (full index) will read regular-boundary face centres at proc-face positions. They MUST use `bm.cf()[facei - nIntF]` (compressed). This mistake is present in at least one commented-out block in `unstructuredMesh.cpp` (line 375) and was a root cause of previous processor-discontinuity bugs.

**`recvBuffer` over-allocation in `communicateBoundaryData`**
- Files: `include/NeoN/fields/boundaryData.hpp`, lines 262, 350
- `recvBuffer` is allocated at `boundaryData.size()` (entire boundary), not `sendSize` (only proc-face count). For a mesh with many physical boundary faces and few proc-face patches, this wastes device memory and may cause an off-bounds read when the MPI receive count is smaller than the allocated buffer.

**`MPI_Alltoallv` send/recv counts are symmetric (same array used for both)**
- Files: `src/mesh/unstructured/unstructuredMesh.cpp` line 491, `include/NeoN/fields/boundaryData.hpp` lines 270–280
- Both use `commPattern.sendCounts` for both `sendcounts` and `recvcounts` arguments of `MPI_Alltoallv`. This is only correct if the decomposition is symmetric (rank A sends the same number of faces to rank B as B sends to A). On non-uniform decompositions or with empty patches this will produce MPI data corruption or hang.

**`boundaryMapVector` in `CommunicationPattern` is always empty**
- File: `src/mesh/unstructured/unstructuredMesh.cpp`, line 503–504
- Comment: `// FIXME seems unused`. The `boundaryMapVector` field of `CommunicationPattern` is populated as an empty `std::vector<localIdx>`. If any downstream code (e.g. Ginkgo distributed solver setup) relies on it for indexing, it will either crash or produce incorrect column indices.

**`create1DUniformMeshPart` only supports 3 ranks**
- File: `src/mesh/unstructured/unstructuredMesh.cpp` (via `include/NeoN/distributed/partitioning.hpp`)
- The helper for constructing a 1D distributed test mesh is the only mesh-partitioning path available. It has hard-coded rank-count assumptions. There is no general N-rank mesh decomposition path — testing on 2, 4, or 8 ranks is not supported by this utility.

**GPU fence scope is too broad**
- File: `include/NeoN/fields/boundaryData.hpp`, line 268
- `fence(boundaryData.exec())` synchronises the entire device before each MPI send. On GPU builds this stalls the full device, preventing async overlap between compute and communication. A stream-specific fence or a Kokkos fence on the specific view would be correct.

---

## Priority Items

1. **`std::abs` on GPU in `updateWeights` proc-face kernel** (`basicGeometryScheme.cpp:215`) — will cause build failure on GPU targets. Replace with `Kokkos::abs` or a device-annotated helper.

2. **`SurfaceField` internalVector/boundaryData desync** — proc-face slots of `internalVector()` are set to the local extrapolation, not the received ghost value. Any operator reading `internalVector()[nIntF+nBnd..]` for a proc face computes with stale data.

3. **`faceToMatrixAddress` offsets incorrect in distributed mode** (`faceToMatrixAddress.hpp:127`) — `diagIdx()` / `upperIdx()` / `lowerIdx()` give wrong flat-index values for proc-adjacent cells. This corrupts all implicit operator assembly at proc boundaries.

4. **`DiagonalSolver::solveDist` is a silent no-op** (`diagonalSolver.hpp:28–38`) — returns empty stats and leaves `x` unchanged. Any distributed PISO/SIMPLE iteration that falls through to this solver produces a zero solution with no error or warning.

5. **`MPI_Alltoallv` symmetric send/recv-count assumption** (`unstructuredMesh.cpp:491`, `boundaryData.hpp:270`) — breaks correctness on non-uniform decompositions. Must use separate `sendCounts` and `recvCounts` arrays.

6. **Communication pattern recomputed per `correctBoundaryConditions` call** (`volumeField.cpp:127`, `surfaceField.cpp:49`) — performance bottleneck on every time step. Cache the `CommunicationPattern` in the field or mesh object.

7. **`partitioning.hpp` 3-rank hard-code** — blocks any distributed test beyond 3 ranks. Needs a generalised N-rank partitioning utility before the distributed stack can be meaningfully benchmarked.

8. **`ginkgoDistributed.cpp` matrix/solver recreated every solve** (`ginkgoDistributed.cpp:105,236`) — major performance regression on iterative time-stepping. Solver factory and matrix assembly should be cached and only invalidated on topology change.

9. **`deltaCoeffs` proc-face asymmetry** (`basicGeometryScheme.cpp:276–295`) — Laplacian coefficient is wrong on non-uniform grids; produces asymmetric matrix that may degrade convergence or produce incorrect solutions.

10. **`recvBuffer` over-allocation and potential OOB read** (`boundaryData.hpp:262,350`) — `recvBuffer` sized at full boundary, not proc-face count. Fix sizing to match actual receive count.

---

*Concerns audit: 2026-05-10*
