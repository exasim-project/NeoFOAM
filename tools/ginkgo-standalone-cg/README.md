# Standalone Ginkgo distributed CG reproducer (T4)

This is the **track T4** isolation harness from the GPU multi-rank proc-boundary divergence
debug session (see `.planning/debug/gpu-proc-boundary-divergence.md` and the SPUMA audit at
`.planning/debug/spuma-comparison.md`).

## What it does

It reads a per-rank dump tree produced by an instrumented NeoFOAM `neoIcoFoam` run with
`NEOFOAM_FULL_DUMP=1` (see `include/NeoFOAM/auxiliary/fullDump.hpp::dumpDistLinearSystem`),
rebuilds the *exact rank-local* assembled `LinearSystem`, and replays it through Ginkgo's CG
solver twice:

1. on `gko::ReferenceExecutor::create()` (CPU baseline);
2. on `gko::CudaExecutor::create(0, ...)` (the same backend NeoN uses for its GPU run).

It then reports `L∞(x_cpu - x_cuda)` per rank and the global max.

The interpretation:

| Outcome | Meaning | Next action |
|---|---|---|
| `L∞ < 1e-10` everywhere | Ginkgo CPU and CUDA agree on the dumped `A, b` — Ginkgo is **innocent**. | Bug is in NeoN's NeoN→Ginkgo composition (stream sharing / fence coverage / halo exchange). Focus on Option C from the SPUMA audit and the host-stage MPI fence path. |
| `L∞` non-trivially large | Ginkgo CPU and CUDA **disagree** on identical inputs. | Ginkgo distributed-CUDA bug. Package the dump as an upstream reproducer and file against `ginkgo-project/ginkgo`. |

## Prerequisites

- A C++20 compiler matching NeoN (GCC ≥ 12, Clang ≥ 15).
- Ginkgo 2.0.0 installed and discoverable, matching the pin in
  `src/NeoN/cmake/Versions.cmake` (`6a3abf8c920228006f3b28bc3bf04fc7a5f6aee0`).
  On HPC the easiest path is to point at NeoN's existing build:
  ```bash
  export Ginkgo_DIR=/path/to/NeoFOAM/build/develop/_deps/ginkgo-build
  ```
  …or whichever directory exports `GinkgoConfig.cmake`.
- MPI 3.1+ (OpenMPI / MPICH).
- CUDA toolkit on the build host (so Ginkgo can compile its CUDA backend).

## Build

```bash
cd tools/ginkgo-standalone-cg
./build.sh                       # build/standalone_cg
# or with explicit ginkgo path:
GINKGO_DIR=/scratch/ginkgo/build ./build.sh
```

## Generate a dump tree

On HPC, in a decomposed case:

```bash
source /path/to/OpenFOAM-2406/etc/bashrc
cd /path/to/decomposed/case             # processor0/, processor1/, ...
env NEOFOAM_FULL_DUMP=1 NEOFOAM_PROC_DUMP=1 \
    UCX_TLS=tcp \
    mpirun -np 2 /path/to/build/develop/bin/neoIcoFoam -parallel
ls processor0/dumps/                     # one file per (step, piso, nonOrth, ckpt, name, kind)
```

The dumps produced by the LinearSystem hook are named:
- `<prefix>__<name>_A_rowptr.txt` / `_A_colidx.txt` / `_A_values.txt`
- `<prefix>__<name>_nonLocalA_*.txt`
- `<prefix>__<name>_b.txt` / `_x0.txt`
- `<prefix>__<name>_partition.txt`

For example:
- `0001_0_0_before_UEqn_solve__U_A_values.txt`   — assembled U-momentum CSR
- `0001_0_0_before_pEqn_solve__p_A_values.txt`   — assembled pressure CSR

## Run

```bash
# from this directory, after build.sh
./run.sh /path/to/case/dump_root 0001_0_0_before_pEqn_solve__p
# or:
mpirun -np 2 build/standalone_cg /path/to/case/dump_root 0001_0_0_before_pEqn_solve__p
```

The "dump root" is the directory containing `processor0/`, `processor1/`, ..., each with a
`dumps/` subdirectory.

The `<checkpoint>` argument is the file basename **without** the `_<kind>.txt` suffix
(e.g. `0001_0_0_before_pEqn_solve__p` — the tool appends `_A_rowptr.txt`, `_A_colidx.txt`, etc.).

Optional arguments:
```
./run.sh <dump_dir> <checkpoint> [nranks=2] [iters=200] [tol=1e-12]
```

## Output

```
[rank 0] nRows=75  CPU(iters=12, |r|=1.234e-13)  CUDA(iters=12, |r|=1.234e-13)  L_inf(x_cpu-x_cuda)=2.220e-16
[rank 1] nRows=75  CPU(iters=12, |r|=1.234e-13)  CUDA(iters=12, |r|=1.234e-13)  L_inf(x_cpu-x_cuda)=2.220e-16
GLOBAL L_inf(x_cpu - x_cuda) over all ranks = 2.220e-16
VERDICT: Ginkgo CPU and CUDA AGREE on the dumped A,b — Ginkgo is INNOCENT for this LinearSystem.
```

## Limitations / scope

- This tool replays **rank-local** CSR blocks independently — it does **not** rebuild the
  distributed `gko::experimental::distributed::Matrix` from the `nonLocalA_*` COO dumps.
  Per-rank-local replay is sufficient for the immediate question "does Ginkgo CUDA produce
  the same x as Ginkgo CPU on the SAME A, b that NeoN handed it?". If that test PASSES the
  bug is provably in the NeoN-side coupling (halo, stream, fence) and not in Ginkgo.
- Vec3-valued systems (the U-momentum solve) currently extract only the first component.
  The diagnostic value is the same — divergence at component 0 implies divergence elsewhere.
- The tool does NOT diff CPU-NeoFOAM-assembled `A, b` against GPU-NeoFOAM-assembled `A, b`
  directly — that's a second mode worth adding as a stretch goal. For now, run the toy case
  twice (CPU executor and GPU executor), and `diff` the dumped `_A_values.txt` /
  `_b.txt` files manually — if they differ, the bug is upstream of Ginkgo in NeoN's
  GPU assembly path.

## Files

- `CMakeLists.txt` — CMake project (finds Ginkgo + MPI, builds `standalone_cg`).
- `main.cpp` — entry point. Loads dump, runs CG twice, reports L∞.
- `build.sh` — wraps `cmake -S . -B build && cmake --build build -j`.
- `run.sh` — wraps `mpirun -np <N> build/standalone_cg ...`.

## Reference

- Track T4 in `.planning/debug/gpu-proc-boundary-divergence.md` (2026-05-18 entry).
- Dump producer: `include/NeoFOAM/auxiliary/fullDump.hpp::dumpDistLinearSystem`.
- SPUMA structural audit motivating the test: `.planning/debug/spuma-comparison.md`
  (section "F1 — Suspect #1 (Ginkgo CG cross-stream visibility)").
