<!--
SPDX-License-Identifier: GPL-3.0-or-later
SPDX-FileCopyrightText: 2026 NeoFOAM authors
-->

# Cylinder benchmark case (Spec 04)

Self-contained flow-past-a-cylinder case driven by
[`../test_cylinder_runtime.py`](../test_cylinder_runtime.py) for the cross-solver
runtime comparison (**PERF-1**) and two grid studies. Both solvers run **genuine
3D** meshes with **roughly cubic cells** at matched physics
(`Re = U·D/ν = 20`, `D = 0.2`, `ν = 0.01`, `U = 1`):

| Subcase | Solver | Mesh | Executor |
|---|---|---|---|
| `blockAMR/` | `incompressibleFluidBlockAMR` | Cartesian `2×1×0.8`, cubic, + direct-forcing immersed cylinder (`D=0.2`) | GPU |
| `bodyFitted/` | `incompressibleFluid` (framework PIMPLE) | body-fitted O-grid, **scaled ×10 to `D=0.2`**, extruded to `K` cubic z-layers (free-slip front/back) | CPU |
| `bodyFittedNeoN/` | `incompressibleFluidNeoN` (NeoN/Kokkos-CUDA PIMPLE) | **same `blockMeshDict` as `bodyFitted/`** — pressure via CG + Jacobi (`PCG` + `diagonal`) | GPU |

`bodyFittedNeoN/` shares the body-fitted mesh with `bodyFitted/`, so the two are
the one **true same-mesh CPU-vs-GPU comparison** — identical discretisation, only
the backend differs. It selects the device via `controlDict` `executor GPU;` (the
framework NeoN solver now threads that key through to `create_adapter_run_time`,
which defaults to Serial). NeoN has no `slip`/`corrected` schemes, so the z-faces
use `zeroGradient` (equivalent for the `z`-homogeneous slab) and the laplacian is
`uncorrected`. It runs only when a CUDA device is present.

Both meshes are homogeneous in `z` (the wake is a 2D flow); `z` is a knob for
mesh size, not new physics. Cells are kept **cubic** on both:

- **blockAMR** — `nz = nx·Lz/Lx` so `dz = dx`. This matters: the nodal MLMG
  coarsens well on cubic cells but **diverges** on anisotropic ones (`dx ≫ dz`).
- **bodyFitted** — extruded to `K ≈ Lz/dx` layers so the bulk cells are cubic.
  (Near-wall boundary-layer cells are graded thin by design — cubic applies to
  the bulk.) The ×10 scale makes its absolute `dx` comparable to blockAMR's,
  which is what the matched-cell-size study needs.

## The three studies

1. **Cell-count scaling** (`test_cell_count_scaling`) — each solver sized to a
   sequence of total-cell targets (`0.1M → 4M`); per-step time + throughput
   (`ms/Mcell`) vs cells. Shows the GPU leaving its starved regime as the mesh
   grows.
2. **Matched cell size** (`test_matched_cell_size`) — all solvers at the same
   absolute `dx` per level, side-by-side. Cell counts differ (the domains
   differ: blockAMR box volume ≈ 4× the body-fitted one), so this is the fair
   same-resolution cost comparison.
3. **Box-size sweep** (`test_max_size_sweep`, blockAMR only) — fix a mesh and
   sweep the AMReX `max_grid_size` (`meshDict` `maxSize`), the box-decomposition
   knob, at a realistic MLMG tolerance (`rtol=1e-4`). **Finding:** box
   decomposition *does* converge — the single-box default `rtol` of `1e-10` is
   what's unreachable once the domain is split (the cross-box coarse-grid
   correction stalls at ~`1e-5`). But on **one GPU it only adds overhead**:
   single-box is fastest (e.g. 44 ms/step at 819k), 2-box ~1.8×, 16-box ~5×, and
   some decompositions still stall depending on box coarsenability (`blockingFactor`
   aligns the box sizes, a standard AMReX knob). Box splitting is an MPI/multi-GPU
   tool, not a single-GPU throughput knob.

Results are written to `benchmarks/results/*.csv` (one tidy row per solver per
level) alongside the console tables.

## What the harness rewrites

The case files carry defaults; the benchmark overrides per run so every solver
does the **same number of fixed steps** with `adjustTimeStep` off:

- `blockAMR/system/meshDict` → `nCell` (cubic `nz = nx·Lz/Lx`), optional
  `maxSize` (box-size sweep); `blockAMR/system/controlDict` → `deltaT`
  (`CFL=0.2`), `endTime`, `executor`.
- `bodyFitted/` and `bodyFittedNeoN/` `system/blockMeshDict` → per-block in-plane
  counts (×`f`) and z-layer count (`K`); `.../system/controlDict` → `deltaT`,
  `endTime` (+ `writeInterval` for the OpenFOAM leg's completion check).
  `blockMesh` then builds the mesh.

## Run

```bash
source ~/OpenFOAM/OpenFOAM-v2406/etc/bashrc   # ABI-matched OpenFOAM
pytest benchmarks/test_cylinder_runtime.py -s
```

Takes ~20-30 min (the 2M/4M legs dominate). Not part of the default test run
(`benchmarks/` is outside `testpaths`). Env overrides: `NEOFOAM_BENCH_TARGETS`
(cell-count list), `NEOFOAM_BENCH_DX` (dx list), `NEOFOAM_BENCH_MAXSIZE_NX` /
`NEOFOAM_BENCH_MAXSIZE` / `NEOFOAM_BENCH_MAXSIZE_RTOL` (box-size sweep),
`NEOFOAM_BENCH_WARMUP` / `NEOFOAM_BENCH_STEPS`, `NEOFOAM_BENCH_CSV_DIR` (results
directory).

### Debugging the nodal solve

blockAMR MLMG verbosity is a **config field** in the `fvSolution` `blockAMR`
subdict — `verbose` (per-iteration nodal-solve residuals) and `bottomVerbose`
(bottom-solver residuals), both `0` (quiet) by default:

```c++
blockAMR
{
    ...
    verbose         2;   // AMReX MLMG residual trace
    bottomVerbose   2;   // + bottom-solver residuals
}
```

This is how the box-size stall (multi-box MLMG plateauing at ~`1e-5`) was
diagnosed.
