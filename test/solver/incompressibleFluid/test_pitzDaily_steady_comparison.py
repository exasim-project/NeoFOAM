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

Tolerance. Each field is compared by its relative L2 difference
``||framework - native||_2 / ||native||_2`` (over all cells and components),
which must be at most ``1e-9`` after five SIMPLE iterations, written with
``writePrecision 17`` so the output rounding does not decide the result.

* The two runs are not bitwise identical everywhere: pybFoam is compiled at
  install time, ``simpleFoam`` ships prebuilt. On an ``x86-64-v3`` toolchain
  (the ubuntu-26.04 runner's gcc) the pybFoam side is built with FMA, the
  prebuilt OpenFOAM is not, so the last bits of the same arithmetic differ.
* The tutorial's solvers stop at ``relTol 0.1``, and on this case a round-off
  difference grows ~1.6-2x per SIMPLE iteration. After fifty iterations FMA
  noise is as large as a real 1e-6 change to the algorithm under every metric;
  tighter linear solvers do not stop the growth. After five iterations the two
  separate cleanly.
* At five iterations the FMA difference is at most 1.5e-11 (65x below the
  bound). A 1e-6 relative change to the U relaxation factor (applied to the
  framework run only) raises the largest field's difference to 3.1e-6, the
  same change to ``nu`` to 6.9e-7 (epsilon) — 690x above the bound or more.
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

# A fixed budget of five SIMPLE iterations (deltaT 1 -> one outer step per unit
# time), far below what residualControl needs, so both solvers do the same
# number of outer steps. Five, not more: see the tolerance in the docstring.
_END_TIME = 5.0
_WRITE_INTERVAL = 5.0
_WRITE_PRECISION = 17

# Relative L2 bound per field; justified in the module docstring.
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


def test_framework_matches_native_simpleFoam() -> None:
    """incompressibleFluid vs native simpleFoam on the steady pitzDaily case.

    The framework solver runs the ``SIMPLE`` algorithm; after five outer
    iterations each of its ``U/p/k/epsilon`` fields is within a relative L2
    difference of ``1e-9`` of native ``simpleFoam`` (tolerance: module
    docstring). ``residualControl``-driven termination is exercised separately
    by the ``SolutionControl`` unit tests, not here.
    """
    repo_root = Path(__file__).parent.parent.parent.parent
    source_case = repo_root / "tutorials" / "pitzDaily_steady"

    case_framework = repo_root / "test_cases" / "pitzDaily_steady_framework"
    case_native = repo_root / "test_cases" / "pitzDaily_steady_native"

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
