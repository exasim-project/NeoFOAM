# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""PERF-1 — cross-solver 3D cylinder runtime + grid + box-size studies.

Spec 04 (``plans/04-blockAMR-cross-solver-performance-spec.md``). Compares the
laminar incompressible solvers that can run flow-past-a-cylinder, over the
self-contained case in ``benchmarks/cylinder/``. Every leg runs **genuine 3D**
meshes with **roughly cubic cells**, at matched physics (``Re = U·D/ν = 20``,
``D = 0.2``, ``ν = 0.01``, ``U = 1``):

* ``incompressibleFluidBlockAMR`` — Cartesian box ``2×1×0.8`` + direct-forcing
  immersed cylinder, AMReX/JAX projection, **GPU**. ``nz`` scales with ``nx`` so
  ``dz = dx`` (the nodal MLMG coarsens well on cubic cells, diverges on ``dx≫dz``).
* ``incompressibleFluid`` — framework PIMPLE on a body-fitted O-grid (scaled ×10
  to ``D=0.2``), extruded to ``K`` cubic z-layers with free-slip front/back,
  pybFoam/OpenFOAM, **CPU**.
* ``incompressibleFluidNeoN`` — the NeoN/Kokkos-CUDA framework PIMPLE port, on
  the **same body-fitted mesh** as ``incompressibleFluid`` (``benchmarks/cylinder/
  bodyFittedNeoN`` shares its ``blockMeshDict``), pressure solved with **CG +
  Jacobi** (``PCG`` + ``diagonal``), **GPU**. This is the one true same-mesh
  CPU-vs-GPU comparison — identical discretisation, only the backend differs.
  Runs only when a CUDA device is present (``neon.gpu_available()``).

Studies:

``test_cell_count_scaling``
    Each solver sized to a sequence of **total cell-count** targets
    (``0.1M → 4M``); per-step wall-clock and throughput (``ms/Mcell``) vs cells.

``test_matched_cell_size``
    Every solver at the **same absolute cell size** ``dx`` per level. Cell counts
    differ (the domains differ), so this is the fair same-resolution cost.

``test_max_size_sweep``
    blockAMR only: fix a mesh and sweep the AMReX ``max_grid_size`` (meshDict
    ``maxSize``) — the box-decomposition knob — at a realistic MLMG tolerance
    (``MAXSIZE_RTOL``, 1e-4). **Finding:** box decomposition *does* converge (the
    single-box default ``rtol`` of 1e-10 is what's unreachable once split — the
    cross-box coarse-grid correction stalls at ~1e-5), but on **one GPU it only
    adds overhead**: single-box is fastest, more boxes cost more (≈5× at 16
    boxes), and some decompositions still stall depending on box coarsenability
    (where ``blockingFactor`` helps). Box splitting is an MPI/multi-GPU tool, not
    a single-GPU throughput knob.

**Fairness / what is asserted.** A guard, not a microbenchmark. Every run does a
fixed number of steps (``adjustTimeStep`` off) in its own case copy + subprocess.
Per-step cost is isolated from one-time init (JAX/CUDA kernel compilation, mesh
build, warm-up) by a **two-point** measurement: time ``N_WARMUP`` and ``N_TIMED``
steps and take the marginal ``(t1 - t0) / (N1 - N0)``. Hard asserts are only the
**fair, same-backend** ones — each leg completes its work, and per-step time
grows with the mesh within a backend. Times *across* backends / CPU-vs-GPU are
**reported, not asserted**.

Results are written to ``benchmarks/results/*.csv`` (one tidy row per solver per
level) in addition to the console tables.

Not collected by the default test run (pytest ``testpaths = test``). Run with
OpenFOAM sourced and ``-s`` (it takes ~20-30 min — the 2M/4M legs dominate)::

    pytest benchmarks/test_cylinder_runtime.py -s

The knobs are typed module constants (below); override any per-run with the
pytest CLI options in ``benchmarks/conftest.py`` — e.g. ``pytest ... --bench-rtol
1e-6 --bench-targets 100000,800000 --bench-steps 6``. Run ``pytest ... --help``
(under the "cylinder-benchmark" group) for the full list. No environment vars.
"""

import csv
import importlib.util
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest

pytest.importorskip("pybFoam")

CASE_DIR = Path(__file__).parent / "cylinder"

# ---------------------------------------------------------------------------
# Benchmark parameters — typed module constants (the defaults). Override any of
# them per-run through the pytest CLI options declared in ``benchmarks/conftest.py``
# (e.g. ``--bench-rtol 1e-6``, ``--bench-targets 100000,800000``); those set the
# matching attribute on this module before the tests run. No environment vars.
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).parent / "results"

# Two-point step counts: marginal per-step cost = (t[TIMED] - t[WARMUP]) / diff.
N_WARMUP = 3
N_TIMED = 9

# Shared physics / geometry.
U_INF = 1.0
CFL = 0.2  # fixed-step stability over the short run
LX = 2.0  # blockAMR streamwise box length
LZ_BAMR = 0.8  # blockAMR spanwise extent (nz scales with nx to keep dz = dx)

# Body-fitted O-grid, scaled ×10 so D = 0.2 (matches blockAMR). The freestream
# cell is then 3.75e-3 and the (single-layer) z-extent 0.1; extruding to K cubic
# layers gives K ≈ LZ_OF / dx and total cells ≈ OF_INPLANE_BASE · f² · K.
OF_BASE_CELL = 3.75e-3  # freestream cell size at in-plane scale f = 1
OF_LZ = 0.1  # z-extent after the ×10 scale
OF_INPLANE_BASE = 282119  # base in-plane cell count (blockMesh nCells at f=1, K=1)
OF_CUBIC_MODEL = OF_INPLANE_BASE * OF_LZ / OF_BASE_CELL  # cells at f=1, cubic K

# Study 1: total cell-count targets (each solver sized to each).
TARGETS = [100_000, 800_000, 2_000_000, 4_000_000]

# Study 2: matched absolute cell sizes (every solver uses each dx).
DX_LEVELS = [0.05, 0.025, 0.0125]

# Study 3: blockAMR box-size sweep — a fixed mesh (nx → nx·ny·nz cubic) and the
# AMReX max_grid_size (meshDict maxSize) ladder: single-box, then splits.
MAXSIZE_NX = 80
MAXSIZE_LEVELS = [MAXSIZE_NX, MAXSIZE_NX // 2, MAXSIZE_NX // 4]
# Nodal-MLMG relative tolerance for the box-size sweep. The single-box default
# (1e-10) is unreachable once the domain is split — the cross-box coarse-grid
# correction stalls at ~1e-5 — so the sweep uses a realistic CFD tolerance under
# which multi-box grids do converge (still ~4 orders of divergence reduction).
MAXSIZE_RTOL = "1e-4"

# Fair-comparison mode: one common pressure-solve *relative* tolerance for ALL
# three solvers, so their runtimes are measured at the same convergence. ``None``
# keeps each case's native tolerances. When set, blockAMR uses it as the nodal
# MLMG ``rtol`` and the OpenFOAM / NeoN pressure solves use it as ``relTol`` (with
# the absolute floor lowered so the relative criterion is the one that binds).
# The algorithms still differ (blockAMR = one Chorin projection/step; OF/NeoN =
# PIMPLE with nCorrectors pressure solves), so this equalises the solve target,
# not the number of solves.
BENCH_RTOL: Optional[str] = None

# Append rows across pytest invocations instead of starting a fresh CSV — for
# gathering a long study in scoped chunks. Rows are always flushed per-measurement.
CSV_APPEND = False


def _gpu_total_bytes() -> int:
    """Total device memory in bytes (for splitting it between JAX and AMReX)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
        return int(out.splitlines()[0].strip()) * 1024 * 1024  # MiB → bytes
    except Exception:
        return 12 * 1024**3


# blockAMR shares the GPU between two allocators that must TOGETHER stay under
# 100%: JAX (XLA) preallocates MEM_FRACTION, AMReX preallocates its arena. Giving
# JAX ~half and AMReX ~a third (≈85% total, 15% headroom) keeps neither starved.
# On-demand AMReX growth (arena 0) or an over-100% split causes OOM/contention and
# spurious MLMG aborts at large meshes. Coarsenable dims (BAMR_ALIGN) keep the
# actual footprint modest, so a single fixed split works across the whole sweep.
_BLOCKAMR_ENV = {
    "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
    "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.5",
    "AMREX_THE_ARENA_INIT_SIZE": str(int(0.35 * _gpu_total_bytes())),
}

_BLOCKAMR_DRIVER = """
import sys, time
import numpy as np
from neofoam.solver.incompressibleFluidBlockAMR import run
t0 = time.perf_counter()
ctx = run(["incompressibleFluidBlockAMR"])
dt = time.perf_counter() - t0
eng = ctx.models["projection_state"]
maxu = float(np.max(np.abs(np.asarray(eng.U.mf[0].arrays()[0]))))
sys.stdout.write(f"BENCH_SECONDS={dt:.4f}\\n")
sys.stdout.write(f"BENCH_MAXU={maxu:.6f}\\n")
sys.stdout.flush()
"""

_INCOMPRESSIBLE_FLUID_DRIVER = """
import time
from neofoam.solver.incompressibleFluid import run
t0 = time.perf_counter()
run(["incompressibleFluid"])
print(f"BENCH_SECONDS={time.perf_counter() - t0:.4f}", flush=True)
"""

# NeoN leg. os._exit(0) after timing skips Python/Kokkos finalization: the
# MeshAdapter otherwise destructs after Kokkos::finalize and aborts (SIGABRT) —
# the same never-finalize teardown as AMReX. BENCH_SECONDS is flushed first.
_NEON_DRIVER = """
import os, sys, time
from neofoam.solver.incompressibleFluidNeoN import run
t0 = time.perf_counter()
run(["incompressibleFluidNeoN"])
sys.stdout.write(f"BENCH_SECONDS={time.perf_counter() - t0:.4f}\\n")
sys.stdout.flush()
os._exit(0)
"""


# ---------------------------------------------------------------------------
# Sizing: map a target cell-count / cell-size to per-solver mesh parameters.
# ---------------------------------------------------------------------------


# Align every blockAMR cell-count dimension to a multiple of this. 16 = 2⁴
# guarantees the AMReX nodal MLMG can coarsen ≥4 levels (real multigrid) whatever
# the target; 32 / 64 give even deeper coarsening. This is the single most
# important blockAMR performance knob: NON-coarsenable dims (e.g. odd 215×107)
# collapse the multigrid to ONE level → the pressure solve turns ~10-25× slower
# per cell (measured: 215/1.98M = 6100 ms/step vs coarsenable 200/1.6M = 182).
BAMR_ALIGN = 16


def _round_align(x: float) -> int:
    """Round to a multiple of ``BAMR_ALIGN`` (≥ BAMR_ALIGN) — a coarsenable count."""
    return max(BAMR_ALIGN, int(round(x / BAMR_ALIGN)) * BAMR_ALIGN)


def _bamr_ny_nz(nx: int) -> tuple[int, int]:
    """The (ny, nz) that keep cells ~cubic for the 2×1×0.8 domain, each aligned
    to a multiple of ``BAMR_ALIGN`` so all three dimensions coarsen."""
    return _round_align(nx / 2.0), _round_align(nx * LZ_BAMR / LX)


def _bamr_nx_for_cells(cells: float) -> int:
    """blockAMR nx giving ~``cells`` cubic cells (cells ≈ 0.2·nx³), coarsenable."""
    return _round_align((cells / 0.2) ** (1.0 / 3.0))


def _bamr_nx_for_dx(dx: float) -> int:
    """blockAMR nx giving cell size ~``dx`` (dx = LX/nx), coarsenable."""
    return _round_align(LX / dx)


def _of_fk_for_cells(cells: float) -> tuple[float, int]:
    """Body-fitted (in-plane scale f, cubic z-layers K) for ~``cells`` cells."""
    f = (cells / OF_CUBIC_MODEL) ** (1.0 / 3.0)
    return f, max(2, round(OF_LZ / OF_BASE_CELL * f))


def _of_fk_for_dx(dx: float) -> tuple[float, int]:
    """Body-fitted (f, K) giving freestream cell size ~``dx`` (cubic z)."""
    return OF_BASE_CELL / dx, max(2, round(OF_LZ / dx))


def _neon_available() -> bool:
    """Whether the blockAMR engine (``neon``) can be imported."""
    return importlib.util.find_spec("neon") is not None


def _neon_gpu_available() -> bool:
    """Whether a CUDA device is present for the NeoN GPU leg."""
    try:
        import neon._neon as nn  # type: ignore[import-not-found]  # noqa: PLC0415

        return bool(nn.gpu_available())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# CSV output.
# ---------------------------------------------------------------------------


def _csv_open(filename: str, header: list[str]) -> Path:
    """Start ``RESULTS_DIR/filename`` and return its path.

    Truncates and writes ``header`` for a fresh run. If ``CSV_APPEND`` is set (the
    ``--bench-csv-append`` option) and the file already has content, keeps it and
    appends, so a long run can be gathered across invocations without losing rows.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    append = CSV_APPEND and (path.exists() and path.stat().st_size > 0)
    if not append:
        with path.open("w", newline="") as fh:
            csv.writer(fh).writerow(header)
    return path


def _csv_row(path: Path, row: list[object]) -> None:
    """Append one row and flush — the CSV stays valid if the run is interrupted."""
    with path.open("a", newline="") as fh:
        csv.writer(fh).writerow(row)
        fh.flush()


# ---------------------------------------------------------------------------
# Subprocess runner + case preparation.
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    """One subprocess solve: wall-clock and the parsed marker payload."""

    seconds: float
    stdout: str


def _invoke(case: Path, driver: str, extra_env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """Run a Python solver driver in a fresh interpreter; return the process."""
    proc = subprocess.run(
        [sys.executable, "-c", driver],
        cwd=case,
        capture_output=True,
        text=True,
        timeout=3600,
        env={**os.environ, "FOAM_SIGFPE": "", **extra_env},
    )
    (case / "solver.log").write_text(proc.stdout + proc.stderr)
    return proc


def _run_solver(case: Path, driver: str, extra_env: dict[str, str]) -> RunResult:
    """Run a Python solver in a fresh interpreter; time only the solve.

    Each solver runs in its own process — required for blockAMR / NeoN (AMReX and
    Kokkos may be initialised once per process, and back-to-back in-process runs
    segfault) and clean for OpenFOAM (one ``Foam::Time`` per process).
    """
    proc = _invoke(case, driver, extra_env)
    assert proc.returncode == 0, (
        f"solver in {case.name} failed:\n{proc.stdout[-1500:]}\n{proc.stderr[-2000:]}"
    )
    match = re.search(r"BENCH_SECONDS=([0-9.]+)", proc.stdout)
    assert match, f"no BENCH_SECONDS marker in {case.name} output:\n{proc.stdout[-1500:]}"
    return RunResult(float(match.group(1)), proc.stdout)


def _set_control(control_dict: Path, updates: dict[str, str]) -> None:
    """Rewrite the given ``key value;`` entries in an OpenFOAM/blockAMR dict."""
    lines = []
    for line in control_dict.read_text().splitlines():
        key = line.strip().split(" ")[0] if line.strip() else ""
        if key in updates:
            lines.append(f"{key:<15} {updates.pop(key)};")
        else:
            lines.append(line)
    for key, value in updates.items():  # keys not already present -> append
        lines.append(f"{key:<15} {value};")
    control_dict.write_text("\n".join(lines) + "\n")


def _prepare_blockamr(
    dest: Path,
    n_steps: int,
    nx: int,
    max_size: Optional[int] = None,
    rtol: Optional[str] = None,
) -> tuple[Path, int]:
    """Clone the blockAMR subcase at resolution ``nx`` (cubic) for ``n_steps``.

    ``max_size`` sets the meshDict ``maxSize`` (AMReX max_grid_size); ``None``
    keeps the whole domain a single box. ``rtol`` overrides the fvSolution nodal
    MLMG relative tolerance — needed for multi-box grids, whose cross-box
    coarse-grid correction stalls well before the single-box default (1e-10).
    """
    case = dest
    shutil.copytree(CASE_DIR / "blockAMR", case)
    ny, nz = _bamr_ny_nz(nx)  # each aligned to BAMR_ALIGN so all three coarsen
    # CFL on the smallest cell size (dims are only ~cubic after alignment).
    dt = CFL * min(LX / nx, 1.0 / ny, LZ_BAMR / nz) / U_INF
    mesh_dict = case / "system" / "meshDict"
    text = re.sub(
        r"nCell\s*\(\s*[0-9 ]+\)",
        f"nCell           ( {nx} {ny} {nz} )",
        mesh_dict.read_text(),
    )
    if max_size is not None:
        # Drop any existing maxSize line, then set the requested one.
        text = re.sub(r"^\s*maxSize\s+[0-9]+\s*;\s*$\n?", "", text, flags=re.MULTILINE)
        text = text.replace("periodicity", f"maxSize         {int(max_size)};\nperiodicity", 1)
    mesh_dict.write_text(text)
    if rtol is not None:
        fv = case / "system" / "fvSolution"
        fv.write_text(re.sub(r"rtol\s+\S+;", f"rtol        {rtol};", fv.read_text()))
    _set_control(
        case / "system" / "controlDict",
        {"deltaT": f"{dt:.8g}", "endTime": f"{n_steps * dt:.8g}", "executor": "gpu"},
    )
    return case, nx * ny * nz


def _set_pressure_rtol(fvsolution: Path, rtol: str) -> None:
    """Force the ``p`` / ``pFinal`` solves to a common relative tolerance.

    Sets ``relTol`` = ``rtol`` and drops the absolute ``tolerance`` floor (1e-12)
    inside the pressure blocks only, so the relative criterion is what stops the
    solve — matching how blockAMR's nodal ``rtol`` is applied. Velocity solves are
    left native (blockAMR has no separate momentum linear solve to match).
    """
    text = fvsolution.read_text()

    def fix(match: "re.Match[str]") -> str:
        body = re.sub(r"tolerance\s+\S+;", "tolerance       1e-12;", match.group(2))
        if re.search(r"relTol\s+\S+;", body):
            body = re.sub(r"relTol\s+\S+;", f"relTol          {rtol};", body)
        else:  # pFinal inherits tolerance via $p but sets its own relTol
            body = body.rstrip() + f"\n        relTol          {rtol};\n    "
        return match.group(1) + body + match.group(3)

    for name in ("p", "pFinal"):
        text = re.sub(rf"(\n\s*{name}\s*\n\s*\{{)([^{{}}]*)(\}})", fix, text, count=1)
    fvsolution.write_text(text)


def _prepare_bodyfitted(
    dest: Path,
    n_steps: int,
    f: float,
    k: int,
    source: str = "bodyFitted",
    rtol: Optional[str] = None,
) -> tuple[Path, int, float]:
    """Clone a body-fitted subcase (``source``), scale it (in-plane ×f, ``k``
    cubic z-layers), mesh it, and pin ``n_steps`` fixed steps.

    ``source`` selects ``bodyFitted`` (OpenFOAM CPU) or ``bodyFittedNeoN`` (NeoN
    GPU) — both share the same ``blockMeshDict``, so a given (f, k) yields the
    same mesh. Returns (case, cells, end_time).
    """
    case = dest
    shutil.copytree(CASE_DIR / source, case)
    shutil.copytree(case / "0.orig", case / "0")

    mesh_dict = case / "system" / "blockMeshDict"
    mesh_dict.write_text(
        re.sub(
            r"\(\s*(\d+)\s+(\d+)\s+(\d+)\s*\)\s*simpleGrading",
            lambda m: f"( {max(1, round(int(m.group(1)) * f))} "
            f"{max(1, round(int(m.group(2)) * f))} {k} ) simpleGrading",
            mesh_dict.read_text(),
        )
    )
    log = subprocess.run(
        ["blockMesh", "-case", str(case)],
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    cells_match = re.search(r"nCells:\s*([0-9]+)", log.stdout)
    cells = int(cells_match.group(1)) if cells_match else 0

    if rtol is not None:
        _set_pressure_rtol(case / "system" / "fvSolution", rtol)

    dt = CFL * (OF_BASE_CELL / f) / U_INF
    end_time = n_steps * dt
    updates = {"deltaT": f"{dt:.8g}", "endTime": f"{end_time:.8g}"}
    if source == "bodyFitted":
        # OpenFOAM completion check needs the endTime directory written.
        updates.update({"writeControl": "timeStep", "writeInterval": str(n_steps)})
    _set_control(case / "system" / "controlDict", updates)
    return case, cells, end_time


def _assert_of_completed(case: Path, end_time: float) -> None:
    """The OpenFOAM run only counts if it wrote the endTime directory."""
    time_dirs = [
        float(d.name)
        for d in case.iterdir()
        if d.is_dir() and re.fullmatch(r"[0-9.]+", d.name) and d.name != "0"
    ]
    assert time_dirs, f"{case.name}: no output time dir — solver did not finish"
    assert max(time_dirs) == pytest.approx(end_time, rel=1e-3), (
        f"{case.name}: final time {max(time_dirs)} != endTime {end_time}"
    )


def _assert_neon_completed(stdout: str, n_steps: int) -> None:
    """NeoN writes nothing (os._exit); completion = one ``Time =`` per step, no NaN."""
    steps = len(re.findall(r"^Time = ", stdout, flags=re.MULTILINE))
    assert steps >= n_steps, f"NeoN ran {steps} steps, expected {n_steps}"
    assert not re.search(r"\b(nan|inf)\b", stdout, flags=re.IGNORECASE), (
        "NeoN solve produced nan/inf"
    )


# ---------------------------------------------------------------------------
# Two-point measurement per backend: (total_s at N_TIMED, ms/step, cells).
# ---------------------------------------------------------------------------


def _two_point_blockamr(tmp: Path, nx: int, tag: str) -> tuple[float, float, int]:
    env = _BLOCKAMR_ENV

    case0, _ = _prepare_blockamr(tmp / f"bamr_{tag}_warm", N_WARMUP, nx, rtol=BENCH_RTOL)
    r0 = _run_solver(case0, _BLOCKAMR_DRIVER, env)
    assert float(re.search(r"BENCH_MAXU=([0-9.]+)", r0.stdout).group(1)) < 5.0

    case1, cells = _prepare_blockamr(tmp / f"bamr_{tag}_timed", N_TIMED, nx, rtol=BENCH_RTOL)
    r1 = _run_solver(case1, _BLOCKAMR_DRIVER, env)
    maxu = float(re.search(r"BENCH_MAXU=([0-9.]+)", r1.stdout).group(1))
    assert maxu < 5.0, f"blockAMR {tag} diverged (max|U|={maxu})"

    per_step_ms = 1000.0 * (r1.seconds - r0.seconds) / (N_TIMED - N_WARMUP)
    return r1.seconds, per_step_ms, cells


def _two_point_bodyfitted(tmp: Path, f: float, k: int, tag: str) -> tuple[float, float, int]:
    case0, _, et0 = _prepare_bodyfitted(tmp / f"of_{tag}_warm", N_WARMUP, f, k, rtol=BENCH_RTOL)
    r0 = _run_solver(case0, _INCOMPRESSIBLE_FLUID_DRIVER, {})
    _assert_of_completed(case0, et0)

    case1, cells, et1 = _prepare_bodyfitted(tmp / f"of_{tag}_timed", N_TIMED, f, k, rtol=BENCH_RTOL)
    r1 = _run_solver(case1, _INCOMPRESSIBLE_FLUID_DRIVER, {})
    _assert_of_completed(case1, et1)

    per_step_ms = 1000.0 * (r1.seconds - r0.seconds) / (N_TIMED - N_WARMUP)
    return r1.seconds, per_step_ms, cells


def _two_point_neon(tmp: Path, f: float, k: int, tag: str) -> tuple[float, float, int]:
    case0, _, _ = _prepare_bodyfitted(
        tmp / f"neon_{tag}_warm",
        N_WARMUP,
        f,
        k,
        source="bodyFittedNeoN",
        rtol=BENCH_RTOL,
    )
    r0 = _run_solver(case0, _NEON_DRIVER, {})
    _assert_neon_completed(r0.stdout, N_WARMUP)

    case1, cells, _ = _prepare_bodyfitted(
        tmp / f"neon_{tag}_timed",
        N_TIMED,
        f,
        k,
        source="bodyFittedNeoN",
        rtol=BENCH_RTOL,
    )
    r1 = _run_solver(case1, _NEON_DRIVER, {})
    _assert_neon_completed(r1.stdout, N_TIMED)

    per_step_ms = 1000.0 * (r1.seconds - r0.seconds) / (N_TIMED - N_WARMUP)
    return r1.seconds, per_step_ms, cells


def _neon_point(tmp: Path, f: float, k: int, tag: str) -> Optional[tuple[float, int]]:
    """NeoN two-point, tolerant of large-mesh failure (CG+Jacobi is a weak
    preconditioner — huge iteration counts / GPU OOM are possible at 2M/4M).
    Returns (ms/step, cells) or None if the run did not complete."""
    try:
        _, ms, c = _two_point_neon(tmp, f, k, tag)
        return ms, c
    except (AssertionError, subprocess.TimeoutExpired) as exc:
        print(f"  [skip] incompressibleFluidNeoN {tag} did not complete: {str(exc)[:90]}")
        return None


def _fmt_m(cells: int) -> str:
    """Compact cell-count label (e.g. 2000000 -> '2.0M')."""
    return f"{cells / 1e6:.1f}M" if cells >= 1e6 else f"{cells / 1e3:.0f}k"


# ---------------------------------------------------------------------------
# Study 1 — cell-count scaling (all solvers, matched total cells, up to 4M).
# ---------------------------------------------------------------------------


def test_cell_count_scaling(tmp_path: Path) -> None:
    """Per-solver 3D scaling: per-step cost + throughput vs total cell count."""
    have_bamr = _neon_available()
    have_neon = _neon_gpu_available()

    path = _csv_open(
        "cell_count_scaling.csv",
        ["target", "solver", "executor", "cells", "ms_per_step", "ms_per_mcell"],
    )

    # rows: (target, solver, executor, cells, ms/step). Each is written to the CSV
    # as soon as it is measured, so an interrupted run still leaves a valid file.
    rows: list[tuple[int, str, str, int, float]] = []

    def _record(target: int, solver: str, execu: str, cells: int, ms: float) -> None:
        rows.append((target, solver, execu, cells, ms))
        _csv_row(
            path,
            [
                _fmt_m(target),
                solver,
                execu,
                cells,
                round(ms, 2),
                round(1e6 * ms / cells, 2),
            ],
        )

    for target in TARGETS:
        if have_bamr:
            nx = _bamr_nx_for_cells(target)
            _, ms, c = _two_point_blockamr(tmp_path, nx, f"c{target}")
            _record(target, "incompressibleFluidBlockAMR", "gpu", c, ms)
        f, k = _of_fk_for_cells(target)
        _, ms, c = _two_point_bodyfitted(tmp_path, f, k, f"c{target}")
        _record(target, "incompressibleFluid", "cpu", c, ms)
        if have_neon:
            pt = _neon_point(tmp_path, f, k, f"c{target}")
            if pt is not None:
                _record(target, "incompressibleFluidNeoN", "gpu", pt[1], pt[0])

    print(
        f"\n[Study 1] cell-count scaling ({N_TIMED} fixed steps, 3D cubic cells, "
        f"marginal per-step over {N_WARMUP}->{N_TIMED}):"
    )
    if not have_bamr:
        print("  [skip] neon not importable — blockAMR leg skipped")
    if not have_neon:
        print("  [skip] no CUDA device — incompressibleFluidNeoN (gpu) leg skipped")
    print(f"  {'target':>7} {'solver':<30}{'exec':>5}{'cells':>10}{'ms/step':>10}{'ms/Mcell':>10}")
    for t, solver, execu, cells, ms in rows:
        print(
            f"  {_fmt_m(t):>7} {solver:<30}{execu:>5}"
            f"{cells:>10}{ms:>10.1f}{1e6 * ms / cells:>10.2f}"
        )
    print(f"  -> {path}")
    print("  (cross-backend / CPU-vs-GPU times are reported, not asserted)")

    # Fair same-backend guard: refining must cost more per step, for each backend
    # (only meaningful with >1 level).
    for solver in {r[1] for r in rows}:
        pts = [r[4] for r in rows if r[1] == solver]
        assert len(pts) < 2 or pts[-1] > pts[0], f"{solver} per-step did not grow with cells"


# ---------------------------------------------------------------------------
# Study 2 — matched absolute cell size (all solvers, same dx per level).
# ---------------------------------------------------------------------------


def test_matched_cell_size(tmp_path: Path) -> None:
    """All solvers at the same absolute cell size dx; side-by-side per level."""
    have_bamr = _neon_available()
    have_neon = _neon_gpu_available()

    path = _csv_open(
        "matched_cell_size.csv",
        ["dx", "solver", "executor", "cells", "ms_per_step", "ms_per_mcell"],
    )

    rows: list[tuple[float, str, str, int, float]] = []

    def _record(dx: float, solver: str, execu: str, cells: int, ms: float) -> None:
        rows.append((dx, solver, execu, cells, ms))
        _csv_row(path, [dx, solver, execu, cells, round(ms, 2), round(1e6 * ms / cells, 2)])

    for dx in DX_LEVELS:
        if have_bamr:
            nx = _bamr_nx_for_dx(dx)
            _, ms, c = _two_point_blockamr(tmp_path, nx, f"dx{dx}")
            _record(dx, "incompressibleFluidBlockAMR", "gpu", c, ms)
        f, k = _of_fk_for_dx(dx)
        _, ms, c = _two_point_bodyfitted(tmp_path, f, k, f"dx{dx}")
        _record(dx, "incompressibleFluid", "cpu", c, ms)
        if have_neon:
            pt = _neon_point(tmp_path, f, k, f"dx{dx}")
            if pt is not None:
                _record(dx, "incompressibleFluidNeoN", "gpu", pt[1], pt[0])

    print(
        f"\n[Study 2] matched cell size ({N_TIMED} fixed steps, 3D cubic cells; "
        f"cell counts differ because the domains differ):"
    )
    if not have_bamr:
        print("  [skip] neon not importable — blockAMR leg skipped")
    if not have_neon:
        print("  [skip] no CUDA device — incompressibleFluidNeoN (gpu) leg skipped")
    print(f"  {'dx':>8} {'solver':<30}{'exec':>5}{'cells':>11}{'ms/step':>10}")
    for dx, solver, execu, cells, ms in rows:
        print(f"  {dx:>8.4g} {solver:<30}{execu:>5}{cells:>11}{ms:>10.1f}")
    print(f"  -> {path}")
    print("  (matched dx; cross-backend time is reported, not asserted)")

    # Fair same-backend guard: finer dx must cost more per step (only meaningful
    # with >1 level).
    for solver in {r[1] for r in rows}:
        pts = [r[4] for r in rows if r[1] == solver]
        assert len(pts) < 2 or pts[-1] > pts[0], f"{solver} per-step did not grow as dx fell"


# ---------------------------------------------------------------------------
# Study 3 — blockAMR box-size (AMReX max_grid_size) sweep.
# ---------------------------------------------------------------------------


def _boxes(nx: int, ny: int, nz: int, max_size: int) -> int:
    """Number of AMReX boxes the domain is chopped into at ``max_size``."""
    return math.ceil(nx / max_size) * math.ceil(ny / max_size) * math.ceil(nz / max_size)


def test_max_size_sweep(tmp_path: Path) -> None:
    """blockAMR only: performance impact of AMReX max_grid_size (meshDict ``maxSize``).

    At a realistic nodal-MLMG tolerance (``MAXSIZE_RTOL``, default 1e-4) box
    decomposition *does* converge, so this measures its per-step cost. On a
    single GPU splitting the domain only adds overhead (smaller kernels, more MG
    iterations, weaker cross-box coarse-grid correction) with no parallelism to
    gain: single-box is fastest, more boxes cost more, and some decompositions
    still stall (``mlmg-failed``) depending on box coarsenability (where
    ``blockingFactor`` matters). Box splitting is an MPI/multi-GPU tool, not a
    single-GPU throughput knob. Splits are reported, not asserted; the single-box
    baseline is the hard guard.
    """
    if not _neon_available():
        pytest.skip("neon not importable — blockAMR max_size sweep skipped")

    nx = MAXSIZE_NX
    ny, nz = _bamr_ny_nz(nx)
    cells = nx * ny * nz
    env = _BLOCKAMR_ENV

    def _seconds(proc: subprocess.CompletedProcess[str]) -> Optional[float]:
        """Wall-clock from a tolerant run, or None if the solve aborted."""
        if proc.returncode != 0 or "BENCH_SECONDS" not in proc.stdout:
            return None
        return float(re.search(r"BENCH_SECONDS=([0-9.]+)", proc.stdout).group(1))

    path = _csv_open(
        "max_size_sweep.csv",
        [
            "nx",
            "ny",
            "nz",
            "max_size",
            "decomposition",
            "status",
            "ms_per_step",
            "ms_per_mcell",
        ],
    )

    # rows: (max_size, boxes, status, ms/step or None). Both warm and timed runs
    # are tolerant: a split box-array can converge for one step but abort over
    # more, so failure at either point marks the level mlmg-failed.
    rows: list[tuple[int, int, str, Optional[float]]] = []
    for ms_val in MAXSIZE_LEVELS:
        nb = _boxes(nx, ny, nz, ms_val)
        case0, _ = _prepare_blockamr(
            tmp_path / f"ms{ms_val}_warm",
            N_WARMUP,
            nx,
            max_size=ms_val,
            rtol=MAXSIZE_RTOL,
        )
        p0 = _invoke(case0, _BLOCKAMR_DRIVER, env)
        case1, _ = _prepare_blockamr(
            tmp_path / f"ms{ms_val}_timed",
            N_TIMED,
            nx,
            max_size=ms_val,
            rtol=MAXSIZE_RTOL,
        )
        p1 = _invoke(case1, _BLOCKAMR_DRIVER, env)
        s0, s1 = _seconds(p0), _seconds(p1)
        if s0 is None or s1 is None:
            log = p0.stdout + p0.stderr + p1.stdout + p1.stderr
            reason = "mlmg-failed" if "MLMG failed" in log else "failed"
            rows.append((ms_val, nb, reason, None))
            per_step_ms = None
        else:
            per_step_ms = 1000.0 * (s1 - s0) / (N_TIMED - N_WARMUP)
            rows.append((ms_val, nb, "ok", per_step_ms))
        _csv_row(
            path,
            [
                nx,
                ny,
                nz,
                ms_val,
                "single-box" if nb == 1 else f"{nb}-box",
                rows[-1][2],
                round(per_step_ms, 2) if per_step_ms is not None else "",
                round(1e6 * per_step_ms / cells, 2) if per_step_ms is not None else "",
            ],
        )

    print(
        f"\n[Study 3] blockAMR max_grid_size sweep "
        f"(fixed {nx}x{ny}x{nz} = {_fmt_m(cells)} cubic cells, {N_TIMED} steps, "
        f"rtol={MAXSIZE_RTOL}):"
    )
    print(f"  {'maxSize':>8}{'boxes':>8}  {'status':<12}{'ms/step':>10}{'ms/Mcell':>10}")
    for ms_val, nb, status, ms in rows:
        ms_txt = f"{ms:>10.1f}" if ms is not None else f"{'-':>10}"
        mc_txt = f"{1e6 * ms / cells:>10.2f}" if ms is not None else f"{'-':>10}"
        print(f"  {ms_val:>8}{nb:>8}  {status:<12}{ms_txt}{mc_txt}")
    print(f"  -> {path}")
    print(
        "  (finding: box decomposition converges at a realistic rtol but only "
        "adds overhead on one GPU — single-box is fastest; more boxes cost more; "
        "some decompositions still stall. It is an MPI/multi-GPU tool.)"
    )

    # Fair guard: the single-box baseline (largest maxSize) must run, and where a
    # split converges it must not be cheaper than single-box (no single-GPU win).
    single_box = [r for r in rows if r[1] == 1 and r[3] is not None]
    assert single_box, "single-box blockAMR run (maxSize >= max(nCell)) must succeed"
    fastest_single = min(r[3] for r in single_box)
    for ms_val, nb, status, ms in rows:
        if nb > 1 and ms is not None:
            assert ms >= 0.9 * fastest_single, (
                f"maxSize={ms_val} ({nb}-box) unexpectedly faster than single-box"
            )
