# aDIC preconditioner for Ginkgo

A native [Ginkgo](https://github.com/ginkgo-project/ginkgo) implementation of OpenFOAM/SPUMA's
**aDIC** ("approximate DIC") preconditioner — the diagonal-based incomplete-Cholesky preconditioner
whose sequential triangular solves are replaced by one forward and one backward Jacobi-style sweep.

## Why a Ginkgo-native version

NeoN/NeoFOAM already ships an aDIC `gko::LinOp` whose kernels run through **Kokkos**. Because Kokkos
and Ginkgo use different CUDA streams, that version must `Kokkos::fence()` after every apply — a
device-wide synchronisation on every CG iteration, which dominates the cost (aDIC ends up slower than
a plain diagonal preconditioner despite converging in ~half the iterations).

Implementing aDIC **inside Ginkgo's execution model** removes that barrier: the kernels run on the
solver's own executor/stream, ordered with the rest of the CG iteration, no cross-framework fence.

## What's here

| file | contents |
|---|---|
| `adic.hpp` | `gko_adic::Adic<ValueType, IndexType>` — a `gko::EnableLinOp`. Host (reference/OpenMP) kernels inline; CUDA launchers declared. |
| `adic_kernels.cu` | CUDA kernels + launchers (one thread per row). |

### Key idea: gather, not scatter

A CSR matrix stores both `A(i,j)` and `A(j,i)`, and aDIC targets **symmetric** matrices, so every
sweep is a **gather** — each row reads its neighbours and writes only its own entry:

```
generate:  rd[i]   = 1 / ( diag[i] - Σ_{j<i} A(i,j)² / diag[j] )
apply:     x[i]    = rd[i]·b[i]
           work[i] = x[i]    - rd[i]·Σ_{j<i} A(i,j)·x[j]      (forward)
           x[i]    = work[i] - rd[i]·Σ_{j>i} A(i,j)·work[j]   (backward)
```

No atomics, no scatter conflicts — three embarrassingly parallel passes. This was verified to be
numerically identical to SPUMA's scatter-with-atomics formulation (`rd` and apply match to 1e-13).

## Usage

aDIC is symmetric/SPD only. For the OpenFOAM pressure Laplacian (assembled negative-definite), hand
Ginkgo the equivalent `(-A)x = (-b)` (NeoFOAM's `negateSystem` flag already does this).

```cpp
#include "adic.hpp"

auto csr  = gko::share(/* gko::matrix::Csr<double,int> of (-A) */);
auto adic = gko::share(gko_adic::Adic<double, int>::create(exec, csr));

auto cg = gko::solver::Cg<double>::build()
              .with_generated_preconditioner(adic)
              .with_criteria(
                  gko::stop::Iteration::build().with_max_iters(1000u),
                  gko::stop::ResidualNorm<double>::build().with_reduction_factor(1e-6))
              .on(exec)
              ->generate(csr);
cg->apply(b, x);
```

In NeoFOAM this slots straight into `GinkgoSolver::generateInjectedSolver` in place of the Kokkos
`ADICPreconditioner` (both are injected as a `generated_preconditioner`) — the rest of the
`preconditioner aDIC;` / `preconReuse` wiring is unchanged.

### Build

`adic.hpp` is host-only (reference + OpenMP) and compiles with a normal C++ compiler against
Ginkgo's headers. `adic_kernels.cu` must be compiled by `nvcc`; link the two together. Reference/CPU
runs need no CUDA. (HIP: `adic_kernels.cu` ports verbatim — swap the launch syntax and
`CUstream_st` → `GKO_HIP_STREAM_STRUCT`. SYCL: an analogous `parallel_for`.)

## Upstreaming to Ginkgo

This is structured as a self-contained `LinOp` so it drops into an application today. For a proper
Ginkgo PR, the Ginkgo team would refactor it into the modular layout, modelled on
`gko::preconditioner::Ic`:

- `include/ginkgo/core/preconditioner/adic.hpp` — class + `parameters_type`/`Factory` + `parse()`.
- `core/preconditioner/adic.cpp` — `GKO_REGISTER_OPERATION` wiring + the apply driver.
- `core/preconditioner/adic_kernels.hpp` — `GKO_DECLARE_*` kernel declarations.
- `reference/`, `omp/`, `common/cuda_hip/`, `dpcpp/` `preconditioner/adic_kernels.*` — the four
  `run()` bodies here become the per-backend kernels (the CUDA/HIP one is shared via
  `common/cuda_hip`).
- Register `"preconditioner::Adic"` in `core/config/registry.cpp` so it is selectable by name.
- Multi-RHS: loop the Dense columns (this version assumes a single, stride-1 RHS).

A CPM patch (`PATCH_COMMAND` on the Ginkgo CPM dependency) could inject these into the fetched
Ginkgo source, but that means carrying the modular files + a CMake/registry patch; the self-contained
`LinOp` above is the lighter path until it's merged upstream.
