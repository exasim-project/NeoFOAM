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
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

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

# A small, fixed SIMPLE-iteration budget (deltaT 1 -> one outer step per unit
# time). pitzDaily/kEpsilon needs a few hundred iterations to satisfy the case's
# residualControl, so at 50 iterations neither solver hits residualControl and
# native simpleFoam runs the full budget — both do the same number of outer
# steps, making the field diff a fair like-for-like comparison once the
# framework solver can actually run a steady case.
_END_TIME = 50.0
_WRITE_INTERVAL = 50.0

# The framework solver runs in its own interpreter: on a SIMPLE case it aborts
# via FOAM's ``FatalIOError`` -> ``::exit()``, which would kill the test process
# if run in-process. A subprocess contains the abort as a non-zero return code.
_FRAMEWORK_DRIVER = """
import os
os.environ["FOAM_SIGFPE"] = ""
from neofoam.solver.incompressibleFluid import run
run(["incompressibleFluid"])
"""


def test_framework_matches_native_simpleFoam() -> None:
    """incompressibleFluid vs native simpleFoam on the steady pitzDaily case.

    The framework solver runs the ``SIMPLE`` algorithm and its ``U/p/k/epsilon``
    fields are compared to native ``simpleFoam`` at ``rtol=1e-10/atol=1e-15``.

    The fixed ``_END_TIME=50`` budget validates *algorithm parity*: at 50 outer
    iterations neither solver satisfies the case's ``residualControl``, so both
    do the same number of SIMPLE passes and the comparison is like-for-like.
    ``residualControl``-driven termination is exercised separately by the
    ``SolutionControl`` unit tests, not here.
    """
    repo_root = Path(__file__).parent.parent.parent.parent
    source_case = repo_root / "tutorials" / "pitzDaily_steady"

    case_framework = repo_root / "test_cases" / "pitzDaily_steady_framework"
    case_native = repo_root / "test_cases" / "pitzDaily_steady_native"

    try:
        setup_case(source_case, case_framework, _END_TIME, _WRITE_INTERVAL)
        setup_case(source_case, case_native, _END_TIME, _WRITE_INTERVAL)

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

        all_match, failed_fields, failed_details = compare_solver_fields(
            case_framework, case_native, FIELDS_TO_COMPARE, rtol=1e-10, atol=1e-15
        )
        if not all_match:
            parts = []
            for fname in failed_fields:
                max_abs, max_rel = failed_details.get(fname, (float("nan"), float("nan")))
                parts.append(f"{fname}(abs={max_abs:.3e}, rel={max_rel:.3e})")
            pytest.fail("Field values differ between solvers: " + ", ".join(parts))
    finally:
        for tc in [case_framework, case_native]:
            if tc.exists():
                shutil.rmtree(tc)
