# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Steady-state (simpleFoam) parity for incompressibleFluid on pitzDaily.

This is the steady counterpart to ``test_pitzDaily_comparison.py`` (which covers
the transient ``pimpleFoam`` path). It runs ``incompressibleFluid`` and native
``simpleFoam`` on ``tutorials/pitzDaily_steady`` (``ddtSchemes steadyState``, a
``SIMPLE`` dict with ``residualControl`` + ``relaxationFactors``) and diffs the
final-time fields.

The framework solver runs in a **separate interpreter**: a FOAM ``FatalIOError``
(e.g. a genuine case-setup problem) calls ``::exit()`` and would otherwise tear
down the whole pytest process; in a subprocess it surfaces as a non-zero return
code that the ``returncode == 0`` assertion turns into a normal test failure.

The history behind this test — the SIMPLE gap it originally captured as an
``xfail`` — is described in
``report/incompressibleFluid-steady-state-limitations.md``.

Tolerance: relative L2 at most ``1e-9`` after five SIMPLE iterations at
``writePrecision 17``. Not bitwise: pybFoam is built with FMA on ``x86-64-v3``, the
prebuilt simpleFoam is not, and ``relTol 0.1`` grows that round-off ~2x per
iteration. At five it is at most 1.5e-11; a 1e-6 relative change to the U relaxation
factor or ``nu`` gives 6.9e-7 or more.
"""

import shutil
import subprocess
import sys
from pathlib import Path

from .comparison_helpers import (
    get_time_directories,
    read_internal_fields,
    relative_l2_error,
    setup_case,
)

FIELDS_TO_COMPARE = [
    ("U", "volVectorField"),
    ("p", "volScalarField"),
    ("k", "volScalarField"),
    ("epsilon", "volScalarField"),
]

# Five SIMPLE iterations (deltaT 1), far below residualControl, so both solvers do
# the same number of outer steps; five, not more: see the module docstring.
_END_TIME = 5.0
_WRITE_INTERVAL = 5.0
_WRITE_PRECISION = 17
_REL_L2_TOL = 1e-9

# The framework solver runs in its own interpreter: on a SIMPLE case it aborts
# via FOAM's ``FatalIOError`` -> ``::exit()``, which would kill the test process
# if run in-process. A subprocess contains the abort as a non-zero return code.
_FRAMEWORK_DRIVER = """
import os
os.environ["FOAM_SIGFPE"] = ""
from neofoam.solver.incompressibleFluid import run
run(["incompressibleFluid"])
"""


def test_framework_matches_native_simpleFoam(tmp_path: Path) -> None:
    """incompressibleFluid (SIMPLE) matches native simpleFoam after five iterations."""
    repo_root = Path(__file__).parent.parent.parent.parent
    source_case = repo_root / "tutorials" / "pitzDaily_steady"

    case_framework = tmp_path / "pitzDaily_steady_framework"
    case_native = tmp_path / "pitzDaily_steady_native"

    try:
        for case in (case_framework, case_native):
            setup_case(
                source_case, case, _END_TIME, _WRITE_INTERVAL, write_precision=_WRITE_PRECISION
            )

        framework = subprocess.run(
            [sys.executable, "-c", _FRAMEWORK_DRIVER],
            cwd=case_framework,
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert framework.returncode == 0, (
            "incompressibleFluid aborted on the steady SIMPLE case:\n"
            f"{framework.stdout[-1500:]}\n{framework.stderr[-1500:]}"
        )

        native = subprocess.run(
            ["simpleFoam", "-case", str(case_native)],
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert native.returncode == 0, f"simpleFoam failed: {native.stderr}"

        final_framework = get_time_directories(case_framework)[-1]
        final_native = get_time_directories(case_native)[-1]
        assert final_framework.name == final_native.name == "5", (
            f"final times {final_framework.name} / {final_native.name}, expected 5"
        )
        names = [name for name, _ in FIELDS_TO_COMPARE]
        framework_fields = read_internal_fields(case_framework, final_framework, names)
        native_fields = read_internal_fields(case_native, final_native, names)
        errors = {
            name: relative_l2_error(framework_fields[name], native_fields[name]) for name in names
        }
        for name, error in errors.items():
            print(f"[pitzDaily_steady] {name}: relative L2 = {error:.3e}")
        failures = [f"{n}={e:.3e}" for n, e in errors.items() if not e <= _REL_L2_TOL]
        assert not failures, (
            f"pitzDaily_steady: incompressibleFluid vs simpleFoam relative L2 above "
            f"{_REL_L2_TOL:.0e} for " + ", ".join(failures)
        )
    finally:
        for tc in [case_framework, case_native]:
            if tc.exists():
                shutil.rmtree(tc)
