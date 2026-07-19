# Performance Report — occDrivAerStaticMesh: float mixed-precision

**Date:** 2026-06-28 (evening update)
**Case:** occDrivAerStaticMesh, kOmegaSST, steady SIMPLE, 30 outer iterations
**Execution:** neoSimpleFoam, **4 MPI ranks, host/OpenMP** (Kokkos CUDA space = 0 — CPU profiling build)
**Supersedes for the mixed-precision section:** `PerformanceReport-2026-06-28-full-study.md`
**Scope:** the float mixed-precision suite, which **now works** after three Ginkgo-fork fixes (below), plus the
non-localized float scale-corrected MG preconditioner (`precfloat-sc`).

---

## 1. Headline — float mixed precision is unblocked

Previously every reduced-precision localized run aborted at solver setup. Three fixes (all in
`ginkgo_schwarz_update_matrix_value.patch`) changed that:

1. **`Schwarz::extract_local_matrix`** — convert the fp64 distributed matrix down to float (matching block
   formats Csr+Coo) instead of a strict `gko::as` cast.
2. **`distributed::Matrix::convert_to`/`move_to`** — convert each local block *precision-first in the source
   format*, so a `Coo<double>` off-diagonal becomes `Csr<float>` (fixed the non-localized `Pgm<float>` path).
3. **`Pgm::update_matrix_value`** — dispatch on `system_matrix_`, not the null `obj` of a failed cast (fixed
   the **cached** float path: was crashing on solve 2 with `run_impl … decltype(nullptr)`).

Result (30-step, host; numbers are stable across 3 repeats to ±1 s):

| Variant | mode | result | exec | p-iters | cont |
|---|---|---|---:|---:|---:|
| `mp-double` | fp64 full solve | ✅ | **145 s** | 54 | 2.4e-5 |
| **`precfloat`** | **float MG *preconditioner*** (outer Cg fp64) | ✅ | **144 s** | 53 | 1.6e-5 |
| `mp-float` | full float inner solve (outer fp64 Ir) | ✅ | 164 s | 1 (Ir) | 2.7e-5 |
| `mp-bfloat16` | full bf16 inner solve | ❌ abort | — | — | — |
| `precbf16` | bf16 MG preconditioner | ❌ abort | — | — | — |

---

## 2. The verdict: float *preconditioner* is free; full-float is a loss; bf16 still blocked

- **`precfloat` (float MG preconditioner) = fp64, within noise.** 144 s vs `mp-double` 145 s, **identical
  convergence** (53 iters, continuity 1.6e-5 vs fp64's 1.6e-5; per-solve 1383 ms vs 1413 ms). Running the
  entire localized MG hierarchy in float costs **nothing** in accuracy or host time — it is a clean, validated
  capability. The *speedup* it should deliver (½ the hierarchy's memory traffic) simply doesn't appear on host.
- **`mp-float` (full float inner solve) is ~13 % slower** (164 vs 145 s). The outer fp64 `Ir` refinement
  wrapper adds work with no float payoff on CPU. For host, the **preconditioner-only** float (`precfloat`) is
  strictly the better float mode.
- **bf16 remains blocked** at config dispatch — the distributed Schwarz config uses `value_type_list_base`,
  which excludes bfloat16. A separate, larger Ginkgo change (not addressed here).

---

## 3. Float scale-corrected MG preconditioner — `precfloat-sc`

A new capability that required all three fixes above: **Cg (fp64) + a non-localized, scale-corrected
`Multigrid` preconditioner in float32**, with solver caching (rebuild every 100th solve).

| Run | cache | exec | p-iters | p-ms | cont |
|---|:--:|---:|---:|---:|---:|
| `precfloat-sc` | ✗ | 211.8 s | 13.2 | 2480 | 1.5e-5 |
| **`precfloat-sc-cache-rebuild100`** | ✓ | **175.7 s** | 13.2 | 2545 | 1.6e-5 |

- **Works end-to-end:** 30 steps, 0 errors, converges identically (1.6e-5). Cache confirmed engaging — 1 p
  rebuild (solve 1) then `update_matrix_value` reuse for solves 2–30 (the path the `pgm.cpp` fix unblocked).
- **Best-in-class iteration count: 13.2 iters/solve** (vs 53 localized, 28 non-loc base) — scale correction
  working as intended, and *without* the localized-SC blow-up (that combination was 446 iters / 1709 s).
- **Cache is the big lever here:** 211.8 → 175.7 s = **−17 %**, because the expensive `Pgm` hierarchy is built
  once and refreshed in place instead of regenerated every solve.
- **But slower than localized on host** (175.7 vs 144 s): the scale-corrected non-localized V-cycle costs
  **~1.8× per solve** (2545 vs 1383 ms). The low iteration count doesn't pay that back on CPU.

---

## 4. Profiling — float doesn't move the host profile (and that's the point)

`precfloat` top-down (146 s Kokkos, 30 steps), essentially identical in shape to fp64:

| Region | Time | % | Note |
|---|---:|---:|---|
| `pressureCorrector.pEqn` | 55.8 s | 38 % | MG solve + `solverSetup` 9.9 s — **still dominates** |
| `setup` (one-time) | 32.4 s | 22 % | mesh/decomp/IO |
| `momentumPredictor` | 20.3 s | 14 % | assembly-bound |
| `turbulenceCorrect` | 20.1 s | 14 % | k + ω |

Because CUDA space is unused, the region distribution is unchanged whether the preconditioner is fp64 or
float — the host does the same SpMV/smoother work either way. **The pressure MG solve is the cost (38 %)**, and
that is exactly the region a float hierarchy accelerates *on GPU* (½ bandwidth per V-cycle) and nowhere else.

---

## 5. Comparison to previous runs

| Config | exec | p-iters | notes |
|---|---:|---:|---|
| `precfloat` (localized, float precond) | 144 s | 53 | = fp64, free on host |
| `mp-double` / best-practice (localized fp64) | 145 / 154 s | 53–54 | reference (cross-study variance ~±10 s) |
| `mp-float` (full float) | 164 s | — | +13 % (Ir overhead) |
| `precfloat-sc` (non-loc SC float, cached) | 175.7 s | 13.2 | fewest iters, heavy V-cycle |
| `pMG-localized-ukoSmooth` (fp64, no cache) | 187.7 s | 53 | cache off |
| `pPCG-diagonal` baseline | 242 s | 548 | — |

Net change since the full-study report: **the float column went from "all abort" to "works and matches fp64"**,
and a genuinely new low-iteration float configuration (`precfloat-sc`) is available. No host *speedup* yet —
that is structurally expected (CUDA unused).

---

## 6. Recommended next steps (prioritized, profiling-driven)

1. **GPU is now the critical path — build production + benchmark the float configs.** Every float result here
   is validated for *correctness, convergence and cache reuse* on host, but float's payoff (½ memory traffic on
   the 38 %-of-runtime MG solve, 2× fp32 throughput) is **GPU-only**. Build `productionnvidia_h200`/`h100`
   (they still hold the old patch — reconfigure to pick up all four files) and run **`precfloat` vs
   best-practice fp64 vs `precfloat-sc`** on the H200. This is the single highest-value action; the host study
   has taken the float work as far as it can.
   - Watch the recorded GPU hazards: keep the solver cache in host-only `solveDist` members (cudafe
     `gko::LinOp` bug); ensure `allocator=UmpirePool` is active (else per-temporary `cudaMalloc/cudaFree`).
2. **On GPU, `precfloat-sc` is the interesting bet.** Its profile — **few (13) but heavy, SpMV-bound float
   V-cycles** — is what fp32 accelerates most. If float halves the heavy V-cycle on the H200, its 13-iter
   convergence could close or beat the localized configs; on host it can't show this.
3. **Reuse the Ginkgo CSR sparsity (`createMtx`, ~5–9 %).** Still pending; orthogonal to precision and a free
   host win — extend the cache to a values-only matrix update for p/U/k/ω.
4. **Don't float the uko preconditioner.** Measured headroom is ~0.2 % (uko solves are ~5 % of runtime and use
   a trivial `diagonal` preconditioner converging in 2–4 iters). Not worth it.
5. **bf16 only if a GPU benefit is proven for float first.** It needs `value_type_list()` in the distributed
   Schwarz config + bf16 distributed instantiation — a real Ginkgo change. Defer until float's GPU speedup
   justifies it.

### Status of fixes (this session)
`ginkgo_schwarz_update_matrix_value.patch` now bundles 4 files: `schwarz.cpp/.hpp` (Schwarz float + in-place
update), `matrix.cpp` (block-wise precision conversion), `pgm.cpp` (cached-update null-dispatch fix). All
compile-checked; the `profilingnvidia_h200` binary is rebuilt with them. The production trees need a
reconfigure to apply the full bundle.

---

## 7. NeoN / NeoFOAM code-level bottlenecks — the case is host-overhead-bound

The single most important profiling fact for the libraries: **only 22.9 % of `neoSimpleFoam.timeStep` is spent
in Kokkos kernels** — the other ~77 % is host "remainder" (orchestration, allocation, object construction).
The actual compute kernels (SpMV, gradients) are already efficient; the bottleneck is **per-step host work**.
The precision/GPU recommendations above don't touch this — these do. Sub-region evidence (precfloat, 146 s,
host; `% remainder` = fraction NOT in Kokkos):

| Region | time | % | remainder | what it is |
|---|---:|---:|---:|---|
| `ginkgo.solverSetup` | 10.6 s | 7.2 % | ~100 % | host-side Ginkgo solver/factory construction per solve (120 calls) |
| `ginkgo.createMtx` | 7.5 s | 5.1 % | ~99.7 % | re-marshalling the LinearSystem into a Ginkgo CSR every solve (120 calls) |
| `momentum.construct` | 2.1 s | 1.4 % | ~98.7 % | host-side momentum equation assembly |
| `luw.gradOpCtor` | 1.6 s | 1.1 % | ~98.7 % | linearUpwindV gradient-operator construction (+ **~75 % of host-memory HWM** is its `Vector` allocs) |
| `luw.gradAlloc` / `assemble.explicitSource` | ~1.8 s | 1.2 % | high | per-assemble scratch allocation |

`momentum.assemble` itself (10.3 s) is mostly real compute (67 % Kokkos) — leave it. The targets are the
host-remainder regions above. **On GPU these get worse**, not better: un-pooled allocations become
*synchronizing* `cudaMalloc`/`cudaFree`, and per-step host construction stalls the device.

### NeoN (core library: LinearSystem, Ginkgo interface, Vector, allocator)
1. **Reuse the Ginkgo CSR sparsity — `createGkoMtx*` (~5 %, 99.7 % host).** The matrix structure is invariant
   across SIMPLE iterations; NeoN re-builds the full CSR (symbolic + numeric) into Ginkgo on every solve, ×120.
   Keep the symbolic pattern and do a **values-only update** (mirrors what the preconditioner cache already
   does for the MG hierarchy). Highest-leverage host win, and it directly enables Ginkgo's `update` fast path.
2. **Cut per-step Ginkgo solver-object churn — `solveDist`/`cacheOrUpdateSolver` (~7 %, 100 % host).**
   `solverSetup` is pure host construction. The solver cache helps the MG hierarchy, but factory/`shared_ptr`/
   workspace construction still recurs per solve. Persist the solver + workspace across steps for **all** fields
   (the Layer-A/B persistence work for U should be confirmed complete and applied to k/ω too).
3. **Switch the default allocator to `UmpirePool` and honor `memPoolSize`.** The plain Umpire allocator issues
   raw `cudaMalloc`/`cudaFree` per temporary (synchronizing on GPU); `memPoolSize` is silently ignored unless
   the pool allocator is selected. This is the #1 host-overhead/GPU-sync source for `assemble`.
4. **Pool/persist `Vector` scratch buffers.** Per-component extraction (`getComponent<I>`) and assembly scratch
   allocate fresh `Vector<scalar>` per component/step — `luw.gradOpCtor/Vector` alone owns ~75 % of the
   host-memory high-water mark. Hand persistent buffers into the component/gradient paths.

### NeoFOAM (discretization layer: fvcc operators, deferredCorr, linearUpwindV, gradients)
5. **Persist discretization operators across SIMPLE iterations.** The linearUpwindV gradient operator
   (`gradOpCtor`) and its backing `Vector`s are reconstructed every momentum assemble. The recent fix cut its
   *time* (51 s → 1.6 s) but it is still the dominant *allocation* (~75 % of host HWM) — cache the operator and
   its buffers so neither time nor memory recurs per step.
6. **Fuse the deferred-correction passes — `divlap.deferredCorr` (4.1 s).** It runs `gradCompute` + `limiter` +
   `applyCorr` + `gradAlloc` + `nonOrthCorr` as separate kernels/temporaries; the case is host-launch-bound, so
   fusing passes (and reusing the gradient buffers from #5) cuts both launches and allocations.
7. **Reduce host-serial equation construction — `momentum.construct` (98.7 % host).** Per-step equation
   assembly/marshalling is host glue; trim object construction and redundant copies in the fvcc operator path.
8. **Same CSR-reuse story for turbulence (k/ω).** `createMtx`/`solverSetup` recur for the k and ω solves too
   (part of `turbulenceCorrect`); items 1–2 apply there as well.

### Priority order (by host time × feasibility)
**(1) CSR sparsity reuse [NeoN]** → **(2) UmpirePool + buffer pooling [NeoN]** → **(3) persist gradient
operator + buffers [NeoFOAM]** → **(4) finish solver-object persistence for all fields [NeoN]** → **(5) fuse
deferredCorr passes [NeoFOAM]**. Together these target the ~30–40 s of host remainder per run that no
precision or solver-config change can reach, and they compound with the GPU migration (where the same
allocations turn into device syncs).
