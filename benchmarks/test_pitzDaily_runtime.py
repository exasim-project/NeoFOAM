# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Total-runtime benchmark: incompressibleFluid vs the two pimpleFoam ports.

Runs the bundled ``tutorials/pitzDaily`` case with three solvers and
compares only the total solve wall-clock:

1. ``pimpleFoam``            — the native OpenFOAM C++ binary,
2. ``pimplefoam.py``         — the plain pybFoam Python port
                               (``neofoam.solver.pimplefoam``, no framework),
3. ``incompressibleFluid``   — the framework solver (ModelSpec/operations).

The point is a guard, not a microbenchmark: (3) vs (2) isolates the
framework's dispatch overhead, which must stay small; (2) vs (1) is the
Python-bindings cost, reported for context.

Fairness: every solver runs the same fixed-step work — ``adjustTimeStep``
is forced off (the plain port never adjusts deltaT anyway), and each run
gets its own case copy. Python solve times are measured *inside* the
subprocess (interpreter/import startup excluded); the native binary is
wall-clocked around the process (its startup is milliseconds). Each case
must reach ``endTime`` for its time to count.

Not collected by the default test run (pytest ``testpaths = test``); run
explicitly:

    pytest benchmarks/test_pitzDaily_runtime.py -s

``NEOFOAM_BENCH_STEPS`` overrides the step count (default 200).
"""

import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("pybFoam")

from neofoam.tutorial import clone_case  # noqa: E402

DELTA_T = 1e-4  # pitzDaily tutorial value
N_STEPS = int(os.environ.get("NEOFOAM_BENCH_STEPS", "200"))
END_TIME = N_STEPS * DELTA_T

# The framework must not massively slow the solve relative to the plain
# pybFoam port — same equations, same bindings, only dispatch differs.
MAX_FRAMEWORK_SLOWDOWN = 1.5

_PY_PIMPLEFOAM_DRIVER = """
import time
from neofoam.solver.pimplefoam import PimpleFoam
solver = PimpleFoam(["pimplefoam"])
t0 = time.perf_counter()
solver.run()
print(f"BENCH_SECONDS={time.perf_counter() - t0:.4f}", flush=True)
"""

_INCOMPRESSIBLE_FLUID_DRIVER = """
import time
from neofoam.solver.incompressibleFluid import run
t0 = time.perf_counter()
run(["incompressibleFluid"])
print(f"BENCH_SECONDS={time.perf_counter() - t0:.4f}", flush=True)
"""


def _prepare_case(dest: Path, name: str) -> Path:
    """Clone pitzDaily, mesh it, and pin fixed-step timings."""
    case = clone_case("pitzDaily", dest=dest / name)
    subprocess.run(
        ["blockMesh", "-case", str(case)], check=True, capture_output=True, timeout=60
    )
    control_dict = case / "system" / "controlDict"
    lines = []
    for line in control_dict.read_text().splitlines():
        key = line.strip().split(" ")[0] if line.strip() else ""
        if key == "endTime":
            lines.append(f"endTime         {END_TIME};")
        elif key == "writeInterval":
            lines.append(f"writeInterval   {END_TIME};")
        elif key == "adjustTimeStep":
            lines.append("adjustTimeStep  no;")
        else:
            lines.append(line)
    control_dict.write_text("\n".join(lines))
    return case


def _assert_completed(case: Path) -> None:
    """The run only counts if it reached endTime (same work for all solvers)."""
    time_dirs = [
        d.name
        for d in case.iterdir()
        if d.is_dir() and re.fullmatch(r"[0-9.]+", d.name) and d.name != "0"
    ]
    assert time_dirs, f"{case.name}: no output time directory — solver did not finish"
    assert max(float(t) for t in time_dirs) == pytest.approx(END_TIME), (
        f"{case.name}: final time {time_dirs} != endTime {END_TIME}"
    )


def _run_native(case: Path) -> float:
    log = case / "solver.log"
    with open(log, "w") as f:
        t0 = time.perf_counter()
        result = subprocess.run(
            ["pimpleFoam"], cwd=case, stdout=f, stderr=subprocess.STDOUT, timeout=600
        )
        elapsed = time.perf_counter() - t0
    assert result.returncode == 0, f"pimpleFoam failed:\n{log.read_text()[-2000:]}"
    return elapsed


def _run_python(case: Path, driver: str) -> float:
    """Run a Python solver in its own interpreter; time only the solve."""
    result = subprocess.run(
        [sys.executable, "-c", driver],
        cwd=case,
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ, "FOAM_SIGFPE": ""},
    )
    (case / "solver.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, (
        f"solver in {case.name} failed:\n{result.stdout[-1000:]}\n{result.stderr[-2000:]}"
    )
    match = re.search(r"BENCH_SECONDS=([0-9.]+)", result.stdout)
    assert match, f"no BENCH_SECONDS marker in {case.name} output"
    return float(match.group(1))


def test_pitzDaily_total_runtime(tmp_path: Path) -> None:
    results: dict[str, float] = {}

    case = _prepare_case(tmp_path, "native_pimpleFoam")
    results["pimpleFoam (C++)"] = _run_native(case)
    _assert_completed(case)

    case = _prepare_case(tmp_path, "python_pimplefoam")
    results["pimplefoam.py (pybFoam)"] = _run_python(case, _PY_PIMPLEFOAM_DRIVER)
    _assert_completed(case)

    case = _prepare_case(tmp_path, "incompressibleFluid")
    results["incompressibleFluid (framework)"] = _run_python(
        case, _INCOMPRESSIBLE_FLUID_DRIVER
    )
    _assert_completed(case)

    native = results["pimpleFoam (C++)"]
    print(f"\npitzDaily, {N_STEPS} fixed steps of {DELTA_T}s (endTime {END_TIME}):")
    for name, seconds in results.items():
        print(f"  {name:<34} {seconds:8.2f} s   ({seconds / native:4.2f}x native)")

    framework = results["incompressibleFluid (framework)"]
    plain = results["pimplefoam.py (pybFoam)"]
    assert framework <= plain * MAX_FRAMEWORK_SLOWDOWN, (
        f"framework solver took {framework:.2f}s vs {plain:.2f}s for the plain "
        f"pybFoam port ({framework / plain:.2f}x > {MAX_FRAMEWORK_SLOWDOWN}x limit)"
    )
