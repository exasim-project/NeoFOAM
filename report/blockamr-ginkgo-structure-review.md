# `ginkgo_solve.cpp` — maintainability / SOLID review and restructuring plan

Target: `src/NeoN/src/bindings/blockAMR/ginkgo_solve.cpp` (4296 lines, 168 KB).
Regression gate available: 13 python test files, 57 tests, 3405 lines under
`src/NeoN/test/blockamr/test_ginkgo_*.py`.

The code *works* and the physics/rationale comments are unusually good — this
review is strictly about structure. Nothing below is a correctness bug report.

---

## 0. Where the file stands relative to its neighbours

| file | lines |
|---|---|
| median of the 12 sibling binding files | ~170 |
| `linop.cpp` | 386 |
| `stencil_kernels.cpp` | 1581 |
| `multifab.cpp` | 1788 |
| **`ginkgo_solve.cpp`** | **4296** |

It is 2.4× the largest sibling and ~45% of the whole `blockAMR/` binding
directory. One anonymous namespace spans lines 52–3745 — 98.6% of the file.
Every sibling exports exactly one `registerXxx(nb::module_&)` and keeps helpers
small; this file has grown a complete linear-algebra library inside a binding
translation unit.

**That is the headline finding: this is not a bindings file any more.** It
contains a standalone geometric-multigrid solver (lines 1109–2699, ~1600 lines
— larger than any sibling file on its own) that has nothing to do with Python
binding. Everything else follows from that.

---

## 1. SOLID assessment

### Single Responsibility — violated at file and class level

The one anonymous namespace holds eight independent concerns:

| concern | lines | approx size |
|---|---|---|
| NVTX/wall-clock profiling (`prof`) | 58–138 | 80 |
| flat-vector ↔ MultiFab transfer | 140–289 | 150 |
| MLMG-backed Ginkgo `LinOp`s (`AmrexOp`, `CompositeAmrexOp`, `MlmgPrecond`) | 289–605 | 320 |
| BC parsing + ghost reflection | 607–816 | 210 |
| face-coefficient operator (`FaceCoeffOp`) + fused stencil | 818–1107 | 290 |
| native GMG (kernels + `GmgPrecondT`) | 1109–2699 | **1590** |
| Ginkgo plumbing (executor, logger, criteria, CSR assembly) | 2701–2962 | 260 |
| persistent solvers + nanobind registration | 2964–4296 | 1330 |

`FaceCoeffSolver` (3118–3519, 400 lines) is itself two classes fused: a Krylov
solver and a native stationary GMG solver, selected by a `bool gmgStationary_`
set in the constructor (3181) and branched on in `solve` (3334). Its 20 member
variables are ~14 "only populated when `solver=="gmg"`" (3498–3518, and the
comment says exactly that).

### Open/Closed — closed against every axis it advertises

Six string-dispatch chains, none extensible without editing existing code:

| axis | dispatch site | values |
|---|---|---|
| executor | `makeExecutor` 2704 | reference, cuda |
| Krylov solver | `buildKrylov` 2812–2855 | cg, bicgstab, gmres, ir |
| solver (incl. native) | `FaceCoeffSolver` ctor 3173 | + gmg |
| preconditioner | 3286–3325 | none, mlmg, gmg |
| GMG smoother | `GmgPrecondT` ctor 2176 | rbgs, chebyshev |
| GMG precision | `buildGmgHierarchy` 3367 | fp64, fp32 |

Adding one Krylov solver means touching `buildKrylov`, the two hand-rolled
copies of the same chain in `registerGinkgoSolve` (4055–4077 and 4201–4225),
the docstrings at 3650–3665, and `ginkgo_solve_stub.cpp`. Worse: the
validation for these six axes is scattered across four different constructors
at four different depths, so an invalid combination is rejected at
unpredictable times — some before any allocation (3160), some after the GMG
hierarchy has already been built.

### Liskov — the `PersistentSolver` hierarchy leaks its subclasses

- `PersistentSolver`'s protected constructor takes `allocDense = true` (3055)
  whose sole purpose is to let *one specific subclass variant*
  (`solver=="gmg"`) skip the base's work vectors. The base knows about a
  subclass's configuration.
- `FaceCoeffSolver::solve` (3332) overrides the base and, for one config,
  dispatches to a completely different algorithm that touches none of the
  base's state (`b_`, `x_`, `solver_`, `logger_`, `resLogger_` all stay null).
  A caller holding a `PersistentSolver&` gets a different contract depending on
  a constructor string.

### Interface Segregation — one 24-parameter interface forced on both solvers

`bindPersistent<S>` (3602) hard-codes a single 24-argument constructor
signature for every solver class. `FaceCoeffCsrSolver` therefore carries seven
parameters it cannot use, spelled as commented-out names (3547–3553):

```cpp
int /*gmg_pre_sweeps*/, int /*gmg_post_sweeps*/, ... const std::string& /*gmg_precision*/
```

and then throws at runtime for the combinations it does not support
(3569–3585). The template that was meant to remove duplication is what forces
the fat interface.

### Dependency Inversion — one good abstraction, otherwise concrete throughout

`GmgApplyMf` (2109–2134) is the file's one genuine abstraction: it lets the
stationary solver drive `GmgPrecondT<T>` without knowing `T`. It is exactly the
right idea, applied once. Everywhere else the code depends on concrete
`amrex::MultiFab`, concrete `MLMG`, concrete `gko::matrix::Dense<double>`, and
a `const GmgApplyMf*` raw pointer aliasing a `shared_ptr` held separately
(3514–3515) — an ownership pattern that only works because the two members are
declared adjacently and set in the same function.

---

## 2. Duplication — measured

### 2.1 Device/Host kernel twins: 13 pairs, ~490 lines, bodies identical

| pair | device | host |
|---|---|---|
| `fillDomainBcGhosts*` | 728 | 755 |
| `gmgConvertCopy*` | 1148 | 1164 |
| `gmgConvertAdd*` | 1191 | 1208 |
| `faceCoeffResidScatterNorm*` | 1252 | 1321 |
| `gmgGsColor*` | 1418 | 1471 |
| `gmgRestrict*` | 1533 | 1555 |
| `gmgCoarsenFace*` | 1587 | 1617 |
| `gmgProlongAdd*` | 1651 | 1667 |
| `gmgResidRestrict*` | 1695 | 1754 |
| `gmgChebComputeD*` | 1825 | 1881 |
| `gmgDinvApply*` | 1944 | 1989 |
| `gmgFillChecker*` | 2041 | 2056 |

Diffing a representative pair (`gmgChebComputeDDevice` 1825–1879 vs
`gmgChebComputeDHost` 1881–1941) with indentation normalised produces exactly
three differences: the function name, and the `ParallelFor(vbx, [=] AMREX_GPU_DEVICE(...)`
wrapper being replaced by three nested `for` loops. **The arithmetic is
character-identical.** Every one of these twins is a place where a physics fix
must be applied twice, silently, with no compiler help.

The file already contains the proof that this is avoidable: `gmgNorm2`
(1384–1410) has *no* twin — it uses a single `AMREX_GPU_HOST_DEVICE` lambda and
`amrex::ReduceSum`, and serves both paths.

Two mechanisms exist to collapse the rest:

- **`amrex::Gpu::LaunchSafeGuard` + `HostDeviceParallelFor`.** Verified present
  in the vendored AMReX (`AMReX_GpuControl.H:130`,
  `AMReX_GpuLaunchFunctsG.H:1786`): `HostDeviceParallelFor` dispatches on
  `Gpu::inLaunchRegion()`, and `LaunchSafeGuard(false)` turns the launch region
  off for a scope. Mark the lambdas `AMREX_GPU_HOST_DEVICE`, keep the kernels as
  free functions (the nvcc extended-lambda restriction noted at 723–724 and
  1112–1114 still applies), and open each `apply` with
  `amrex::Gpu::LaunchSafeGuard lsg(onDevice_);`.
- **Or delete the host twins outright.** The reference path already allocates
  every fab in `The_Pinned_Arena` (966–971, 2410–2412, 3235–3240), which is
  device-accessible, so the device kernels are *correct* on that path — just
  slower. Whether "reference" must genuinely execute on the CPU is a decision
  only you can make; if it is purely a correctness oracle, this deletes ~490
  lines for free.

Recommendation: the `LaunchSafeGuard` route, because it preserves the current
"reference means CPU" semantics. It must be verified per kernel (in particular
`ParReduce` at 1305, which has no host branch and would need `ReduceSum` in the
`gmgNorm2` style).

### 2.2 The 7-point stencil formula: written out 13 times

`diag = alpha - (aE+aW+aN+aS+aT+aB)` appears verbatim at lines 880, 1067, 1295,
1366, 1460, 1517, 1742, 1806, 1867, 1927, 1981, 2030, 2927; the matching
six-neighbour off-diagonal sum appears 12 times. This is the file's central
mathematical definition — the `negSumDiag` contract documented at 888–902 — and
it has thirteen independent copies. A change to the discretisation is a
thirteen-site edit.

One `AMREX_GPU_HOST_DEVICE inline` helper pair (`stencilDiag(...)`,
`stencilOffDiag(...)`, or a small `Stencil7` struct returning both) removes all
thirteen. The kernels stay separate — only the arithmetic is shared.

### 2.3 The `(alpha, ux, lx, uy, ly, uz, lz)` bundle

Seven fields travel together as seven positional parameters in five signatures
(827, 1252, 1321, 1418, 1471, 1695, 1754, 1825, 1944, 2863, 3347), as seven
raw member pointers in `FaceCoeffOp` (1098–1104) and again in `FaceCoeffSolver`
(3500–3506), and as seven `shared_ptr`s in `GmgLevelT` (2083). Two different
orderings are in use: constructors take `alpha` first (919–925), the fused
stencil takes it last (827–838). Passing them in the wrong order compiles
cleanly and produces wrong physics.

A `FaceCoeffs<T>` view struct (7 pointers + accessors) collapses all of it into
one named parameter and makes the order a compile-time property of the type.

### 2.4 `apply_impl(alpha, b, beta, x)` — 5 identical copies

Lines 363, 486, 589, 1076, 2383. Every one is:

```cpp
auto denseX = gko::as<Dense>(x);
auto tmp = denseX->clone();
this->apply_impl(b, tmp.get());
denseX->scale(beta);
denseX->add_scaled(alpha, tmp);
```

A CRTP base — `template<class D> class AmrexLinOpBase : public gko::EnableLinOp<D>, public gko::EnableCreateMethod<D>` —
implements it once and all five operators inherit it.

### 2.5 Stopping criteria and result dict — 7 copies each

`gko::stop::ResidualNorm<double>::build()` chains appear at 2801, 2807, 3830,
3836, 4041, 4047, 4194; the `nb::dict{num_iters, res_norm, converged,
res_history}` epilogue at 3037–3047, 3485–3495, 3864–3874, 4114–4124,
4239–4242. Two small free functions (`makeCriteria(exec, max_iter, rtol, atol,
baseline)` and `makeResultDict(logger, resLogger, resNorm)`) remove all of it.

### 2.6 Executor selection — inlined again next to the function that does it

`makeExecutor` exists at 2704. The `ginkgo_solve` lambda re-implements it
inline at 3769–3789, comment and all. One-line fix, listed here only because it
shows how the file's own helpers stopped being discoverable once it passed
~2000 lines.

### 2.7 Coefficient staging — same block twice

`FaceCoeffOp`'s constructor (932–972) and `FaceCoeffSolver`'s gmg branch
(3182–3214) contain the same "device: alias caller pointers / host: seven
`pinnedCopy` calls into `owned_[0..6]`" block. A `FaceCoeffs` type with a
`staged(bool onDevice)` factory absorbs both.

---

## 3. Smaller issues worth fixing while you are in there

- **`#include <nvtx3/nvToolsExt.h>` is unguarded (line 33)** and
  `prof::Timer` calls `nvtxRangePushA`/`nvtxRangePop` unconditionally
  (104, 126). Siblings guard device-specific includes with
  `#if defined(AMREX_USE_CUDA)`. A non-CUDA Ginkgo build will not compile this.
- **`prof::table()`** (86) is a global mutable `std::map`, written from
  `~Timer` with no synchronisation, and `Timer`'s constructor allocates a
  `std::string` key per phase per iteration when profiling is on. Fine for the
  current single-threaded use; worth a comment saying so, or an interned key.
- **Milestone markers as comments** — "M0", "M3 3a", "M4 item 3", "M5",
  "milestone-1 behavior" appear throughout (16, 59, 784, 818, 1234, 2107,
  2654, 3448). They are changelog, not documentation: a reader in six months
  cannot resolve them. The *rationale* they attach to (e.g. the excellent
  two-kernel justification at 1236–1250) should stay; the milestone labels
  belong in `report/blockamr-ginkgo-performance.md`.
- **`makeExecutor` duplicates `NeoN::la::ginkgo::getGkoExecutor`**
  (`include/NeoN/linearAlgebra/ginkgo.hpp:28`), which already memoises per
  executor and registers a finalize hook. Reuse needs `blockAMR/CMakeLists.txt`
  to link `NeoN::NeoN` (it currently links only `amrex` and `Ginkgo::ginkgo`) —
  a deliberate decision, not a free win. Flagging it, not recommending it.
- **`ginkgo_solve_stub.cpp` must mirror the public surface by hand** (6 `m.def`s
  + 2 classes). Any split must keep the surface declared in exactly one place or
  this hazard gets worse.
- **`CMakeLists.txt` lists every source twice** (sources at 12–27, CUDA
  `LANGUAGE` property at 46–65). Splitting this file into ~12 TUs makes that
  duplication actively painful — turn the list into a variable first.
- **Nothing here is testable from C++.** Every path is reachable only through
  nanobind. The GMG hierarchy in particular (1600 lines of numerics) deserves
  direct C++ tests, which requires it to live in a library target rather than a
  binding TU.

---

## 4. Proposed structure

Split into a small static library plus a thin binding TU. The library is
testable from C++ and does not depend on nanobind; the binding TU depends on
both.

```
src/bindings/blockAMR/
  ginkgo/                          # new static lib target, no nanobind
    profiling.hpp/.cpp             prof::Timer, table, NVTX behind a CUDA guard      ~90
    transfer.hpp/.cpp              gather/scatter/scatterShell, Dense<->MultiFab     ~140
    bc.hpp/.cpp                    BcArray, parseBc, BcGhostFill, fillDomainBcGhosts ~140
    face_coeffs.hpp                FaceCoeffs<T> view + HOST_DEVICE stencil helpers  ~130
    linop_base.hpp                 AmrexLinOpBase<D> CRTP (advanced apply_impl)      ~45
    mlmg_ops.hpp/.cpp              AmrexOp, CompositeAmrexOp, MlmgPrecond            ~320
    face_coeff_op.hpp/.cpp         FaceCoeffOp + fused stencil                       ~230
    gmg/kernels.hpp                single-body HOST_DEVICE kernels (twins collapsed) ~480
    gmg/precond.hpp/.cpp           GmgLevelT, GmgApplyMf, GmgPrecondT<T>             ~430
    config.hpp/.cpp                SolverConfig/GmgConfig + all string->enum parsing ~140
    krylov.hpp/.cpp                makeExecutor, makeCriteria, buildKrylov,
                                   ResidualHistoryLogger, makeResultDict             ~210
    csr.hpp/.cpp                   assembleFaceCoeffCsr                              ~120
    solvers.hpp/.cpp               ISolver; KrylovSolver, GmgStationarySolver,
                                   CsrSolver                                         ~430
  ginkgo_solve.cpp                 registrar only: m.def + args + docstrings         ~260
  ginkgo_solve_stub.cpp            unchanged
```

Largest resulting file ~480 lines, in line with `linop.cpp` and well under the
1.6–1.8k of `stencil_kernels.cpp`/`multifab.cpp`.

### Key design changes inside that layout

**a) `FaceCoeffs<T>` replaces the 7-pointer bundle.**

```cpp
template<class T> struct FaceCoeffs {
    const GmgFab<T>* alpha; const GmgFab<T>* ux; /* lx uy ly uz lz */
    AMREX_GPU_HOST_DEVICE Stencil7<T> at(const Array4Set&, int i,int j,int k) const;
};
```
Every kernel signature drops from 9–11 parameters to 3–4, and §2.3's ordering
hazard disappears.

**b) `SolverConfig` replaces the 24-parameter constructor.**

```cpp
struct GmgConfig  { int preSweeps, postSweeps, coarsestSweeps, maxLevels, minBottom;
                    Smoother smoother; Precision precision; };
struct SolverConfig { Executor exec; SolverKind solver; PrecondKind precond;
                      int maxIter; double rtol, atol; bool projectNullspace;
                      BcArray bc; GmgConfig gmg; MLMG* precondMlmg; int precondCycles; };
SolverConfig parseSolverConfig(/* the nanobind kwargs */);   // validates ONCE, at the boundary
```
All six string axes become enums parsed in one place, so an invalid combination
is rejected before any allocation. `bindPersistent` builds the config and hands
it to a constructor; `FaceCoeffCsrSolver` stops carrying seven dead parameters
and instead rejects unsupported *fields* in one `validateForCsr(config)`.

**c) `PersistentSolver` becomes an interface; the GMG variant becomes a class.**

```cpp
struct ISolver { virtual ~ISolver() = default;
                 virtual nb::dict solve(amrex::MultiFab& rhs, amrex::MultiFab& sol) = 0; };

class KrylovSolver        : public ISolver { /* today's PersistentSolver body */ };
class GmgStationarySolver : public ISolver { /* today's gmgSolve + its 14 members */ };
class CsrSolver           : public KrylovSolver { /* CSR assembly only */ };

std::unique_ptr<ISolver> makeFaceCoeffSolver(const FaceCoeffs<double>&,
                                             const amrex::Geometry&, const SolverConfig&);
```
This is the change that fixes the LSP and SRP findings at once: `allocDense`
disappears (the GMG solver simply never had Dense vectors), the
`gmgStationary_` branch in `solve` disappears, and the 14 conditionally-live
members move to the class that owns them. Python still sees one
`FaceCoeffSolver` type — the factory sits behind the binding.

**d) `AmrexLinOpBase<D>` CRTP** removes the five `apply_impl(alpha,b,beta,x)`
copies (§2.4).

**e) `registerGinkgoSolve` keeps only registration.** The three free-function
lambdas (3753–3899, 3900–4153, 4154–4272 — 150–250 lines each) become named
functions in `solvers.cpp`; the `m.def` calls keep the `nb::arg` specs and
docstrings, matching what every sibling file does.

---

## 5. Suggested sequence

Each step is independently shippable and gated on the existing 57 python tests
plus the benchmark timings (several steps touch the hot loop, so a wall-clock
check matters as much as correctness).

| # | step | risk | lines removed |
|---|---|---|---|
| 1 | Guard the NVTX include; move `prof` to `profiling.hpp/.cpp` | none | 0 |
| 2 | Make `ginkgo_solve` call `makeExecutor` (§2.6); extract `makeCriteria` + `makeResultDict` (§2.5) | none | ~120 |
| 3 | `AmrexLinOpBase` CRTP (§2.4) | low | ~40 |
| 4 | Extract `transfer`, `bc`, `csr` to their own TUs — pure moves | low | 0 |
| 5 | Introduce `FaceCoeffs<T>` + `Stencil7` helpers; rewrite kernel signatures and the 13 formula copies (§2.2, §2.3) | **medium — verify bit-identical results** | ~150 |
| 6 | Collapse the 13 device/host twins via `LaunchSafeGuard` (§2.1); one kernel at a time, reference-vs-cuda parity test per kernel | **medium — the payoff step** | ~490 |
| 7 | `SolverConfig` + enum parsing at the boundary | low | ~80 |
| 8 | Split `FaceCoeffSolver` into `KrylovSolver` / `GmgStationarySolver` behind `ISolver` + factory | medium | ~60 |
| 9 | Move GMG to `gmg/`, solvers to `solvers.cpp`, thin out the registrar | low | 0 |
| 10 | CMake: source list as a variable; add the `ginkgo` lib target and C++ tests for the GMG hierarchy | low | 0 |

Net: roughly 4300 → ~2900 lines of actual code, spread over ~13 files none of
which exceeds ~480 lines, with the discretisation defined once instead of
thirteen times.

**Steps 5 and 6 carry the real risk** and deliver most of the value. If you only
do two things, do those two — and do 6 kernel-by-kernel, asserting
reference-vs-cuda parity after each, rather than as one commit.

---

## 6. What not to change

- The comments explaining *why* (the affine-operator derivation at 5–16, the
  composite-operator nullspace discussion at 383–400, the two-kernel
  justification at 1236–1250, the CG-symmetry warnings at 519–526 and
  3154–3159). This is the best-documented file in the directory and the
  restructuring should carry every one of these across intact.
- `GmgApplyMf` — it is the right abstraction and should stay exactly as it is.
- The `gather`/`scatter` flat-index ordering contract (140–149). It is
  load-bearing across host, device and fused paths; moving it to its own TU is
  fine, changing it is not.
