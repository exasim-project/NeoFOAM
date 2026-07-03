# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Field-by-field parity for incompressibleFluid (PIMPLE) on pitzDaily.

Two comparisons on the bundled ``tutorials/pitzDaily``, both with a fixed
time step so every solver does identical stepping:

1. **framework vs the plain pybFoam port** (``neofoam.solver.pimplefoam``) —
   the framework's guarantee: same bindings, same equations, so the fields
   must match to round-off. This guards that the framework machinery
   (operation graph, models, finalIteration handling) does not change the
   physics.
2. **framework vs native ``pimpleFoam``** — round-off parity against the C++
   binary. This used to be ``xfail``: the pybFoam solvers skipped
   ``turbulence->validate()`` (which ``pimpleFoam`` calls before the first
   solve), so the eddy viscosity stayed at the ``0/nut`` placeholder (``0``)
   for the first momentum equation and every field diverged from there. Both
   the framework and the plain port now validate on build, matching native
   bit-for-bit.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from neofoam.solver.incompressibleFluid import run

from .comparison_helpers import (
    compare_solver_fields,
    setup_case,
)

FIELDS_TO_COMPARE = [
    ("U", "volVectorField"),
    ("p", "volScalarField"),
    ("k", "volScalarField"),
    ("epsilon", "volScalarField"),
]

_END_TIME = 0.01
_WRITE_INTERVAL = 0.01

# The plain port runs in its own interpreter: a second in-process solver run
# would construct another Foam::Time next to the framework's.
_PORT_DRIVER = """
import os
os.environ["FOAM_SIGFPE"] = ""
from neofoam.solver.pimplefoam import PimpleFoam
PimpleFoam(["pimplefoam"]).run()
"""

# A ``-parallel`` framework run: each MPI rank is a fresh interpreter spawned by
# mpirun, so the solver drives itself the same way native ``pimpleFoam -parallel``
# does (one ``argList``/``Time`` per rank, all under the same MPI session). The
# driver and the decomposeParDict live in real files rather than inline strings.
_PARALLEL_DRIVER = Path(__file__).parent / "_parallel_driver.py"
_DECOMPOSE_PAR_DICT = Path(__file__).parent / "_parallel_decomposeParDict"

# Two subdomains — the fixture decomposeParDict decomposes into two, keeping the
# test runnable on a two-core CI runner.
_NPROCS = 2


def _mpi_available() -> bool:
    return all(
        shutil.which(tool) is not None
        for tool in ("mpirun", "decomposePar", "reconstructPar")
    )


def _prepare_case(source: Path, target: Path) -> None:
    """Copy + mesh the case and pin a fixed time step (identical stepping).

    The plain port never adjusts deltaT (its CFL criteria list stays empty),
    so comparisons are only meaningful with ``adjustTimeStep no``.
    """
    setup_case(source, target, _END_TIME, _WRITE_INTERVAL)
    control_dict = target / "system" / "controlDict"
    lines = [
        "adjustTimeStep  no;" if line.strip().startswith("adjustTimeStep") else line
        for line in control_dict.read_text().splitlines()
    ]
    control_dict.write_text("\n".join(lines))


def _run_framework(case: Path) -> None:
    original_dir = Path.cwd()
    os.chdir(case)
    try:
        run(["incompressibleFluid"])
    finally:
        os.chdir(original_dir)


def _assert_fields_match(case_a: Path, case_b: Path, rtol: float, atol: float) -> None:
    all_match, failed_fields, failed_details = compare_solver_fields(
        case_a, case_b, FIELDS_TO_COMPARE, rtol=rtol, atol=atol
    )
    if not all_match:
        parts = []
        for fname in failed_fields:
            max_abs, max_rel = failed_details.get(fname, (float("nan"), float("nan")))
            parts.append(f"{fname}(abs={max_abs:.3e}, rel={max_rel:.3e})")
        pytest.fail("Field values differ between solvers: " + ", ".join(parts))


def test_framework_matches_plain_pybfoam_port() -> None:
    """incompressibleFluid must reproduce the framework-free port to round-off."""
    repo_root = Path(__file__).parent.parent.parent.parent
    source_case = repo_root / "tutorials" / "pitzDaily"

    case_framework = repo_root / "test_cases" / "pitzDaily_framework"
    case_port = repo_root / "test_cases" / "pitzDaily_plain_port"

    try:
        _prepare_case(source_case, case_framework)
        _prepare_case(source_case, case_port)

        _run_framework(case_framework)

        env = {**os.environ, "PYTHONPATH": str(repo_root / "src")}
        result = subprocess.run(
            [sys.executable, "-c", _PORT_DRIVER],
            cwd=case_port,
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )
        assert result.returncode == 0, (
            f"plain pybFoam port failed:\n{result.stdout[-1000:]}\n"
            f"{result.stderr[-2000:]}"
        )

        _assert_fields_match(case_framework, case_port, rtol=1e-10, atol=1e-15)
    finally:
        for tc in [case_framework, case_port]:
            if tc.exists():
                shutil.rmtree(tc)


def test_framework_matches_native_pimpleFoam() -> None:
    """The end goal: round-off parity against the C++ binary."""
    repo_root = Path(__file__).parent.parent.parent.parent
    source_case = repo_root / "tutorials" / "pitzDaily"

    case_framework = repo_root / "test_cases" / "pitzDaily_custom_solver"
    case_native = repo_root / "test_cases" / "pitzDaily_native_solver"

    try:
        _prepare_case(source_case, case_framework)
        _prepare_case(source_case, case_native)

        _run_framework(case_framework)

        result = subprocess.run(
            ["pimpleFoam", "-case", str(case_native)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, f"pimpleFoam failed: {result.stderr}"

        _assert_fields_match(case_framework, case_native, rtol=1e-10, atol=1e-15)
    finally:
        for tc in [case_framework, case_native]:
            if tc.exists():
                shutil.rmtree(tc)


def _decompose(case: Path) -> None:
    shutil.copyfile(_DECOMPOSE_PAR_DICT, case / "system" / "decomposeParDict")
    result = subprocess.run(
        ["decomposePar", "-case", str(case), "-force"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"decomposePar failed: {result.stderr}"


def _reconstruct(case: Path) -> None:
    result = subprocess.run(
        ["reconstructPar", "-case", str(case), "-latestTime"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"reconstructPar failed: {result.stderr}"


def _run_framework_parallel(case: Path) -> None:
    repo_root = Path(__file__).parent.parent.parent.parent
    env = {**os.environ, "PYTHONPATH": str(repo_root / "src")}
    result = subprocess.run(
        ["mpirun", "-np", str(_NPROCS), sys.executable, str(_PARALLEL_DRIVER)],
        cwd=case,
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )
    assert result.returncode == 0, (
        f"framework parallel run failed:\n{result.stdout[-1000:]}\n"
        f"{result.stderr[-2000:]}"
    )


def _run_native_parallel(case: Path) -> None:
    result = subprocess.run(
        ["mpirun", "-np", str(_NPROCS), "pimpleFoam", "-case", str(case), "-parallel"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"pimpleFoam -parallel failed: {result.stderr}"


@pytest.mark.skipif(
    not _mpi_available(), reason="mpirun/decomposePar/reconstructPar not available"
)
def test_framework_matches_native_pimpleFoam_parallel() -> None:
    """The ``-parallel`` path must match native ``pimpleFoam`` run in parallel.

    Decomposes pitzDaily identically for both solvers, runs each under
    ``mpirun -np 2 … -parallel``, reconstructs, and compares the fields. Guards
    the framework's parallel wiring: one ``argList``/``Time`` per rank with the
    argList (and the MPI session it owns) kept alive for the whole run, so no
    Pstream exchange happens after MPI is finalized.
    """
    repo_root = Path(__file__).parent.parent.parent.parent
    source_case = repo_root / "tutorials" / "pitzDaily"

    case_framework = repo_root / "test_cases" / "pitzDaily_parallel_framework"
    case_native = repo_root / "test_cases" / "pitzDaily_parallel_native"

    try:
        _prepare_case(source_case, case_framework)
        _prepare_case(source_case, case_native)

        for case, run_parallel in (
            (case_framework, _run_framework_parallel),
            (case_native, _run_native_parallel),
        ):
            _decompose(case)
            run_parallel(case)
            _reconstruct(case)

        _assert_fields_match(case_framework, case_native, rtol=1e-10, atol=1e-15)
    finally:
        for tc in [case_framework, case_native]:
            if tc.exists():
                shutil.rmtree(tc)
