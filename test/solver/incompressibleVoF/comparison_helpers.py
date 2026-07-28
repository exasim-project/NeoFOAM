# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Shared damBreak comparison harness: incompressibleVoF vs a native solver.

Both damBreak comparison tests (MULES vs interFoam, isoAdvector vs
interIsoFoam) run the identical sequence — set up two copies of a damBreak
tutorial, run the Python solver in one and the native OpenFOAM solver in the
other, then compare the written fields — so the whole body lives here and each
test is a one-line parameterization.

``run_dambreak_regime`` is the same set-up/run pair for ``test_mules_regimes``
and ``test_cranknicolson_alpha_ddt``, which need (a) the case dictionaries
varied per run and (b) the two case directories left on disk for several
assertions instead of a single pass/fail, so it stops after the two runs and
lets the caller compare.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Optional, Union

from pybFoam import dictionary

from neofoam.solver.incompressibleVoF import run

# Re-use the comparison helpers from the sibling incompressibleFluid test
# package (``test/solver`` is a package, so a ``..`` relative import reaches it).
from ..incompressibleFluid.comparison_helpers import (
    compare_solver_fields,
    setup_case,
)

# Disable OpenFOAM floating point exception trapping
os.environ["FOAM_SIGFPE"] = ""

# Fields to compare between solvers
FIELDS_TO_COMPARE = [
    ("alpha.water", "volScalarField"),
    ("U", "volVectorField"),
    ("p_rgh", "volScalarField"),
]

_REPO_ROOT = Path(__file__).parent.parent.parent.parent
_SOLVER_WORKER = Path(__file__).parent / "_mules_regime_worker.py"


def run_dambreak_comparison(
    tutorial_name: str,
    native_solver: str,
    case_prefix: str,
    end_time: float = 0.05,
    write_interval: float = 0.05,
) -> None:
    """Run ``tutorials/<tutorial_name>`` with incompressibleVoF and with the
    native ``native_solver`` binary, then assert the written fields match to
    machine precision (rtol=atol=1e-10).

    ``case_prefix`` names the two scratch case directories under
    ``test_cases/`` (``<case_prefix>_incompressibleVoF`` /
    ``<case_prefix>_<native_solver>``), which are removed afterwards.
    """
    repo_root = Path(__file__).parent.parent.parent.parent
    source_case = repo_root / "tutorials" / tutorial_name

    test_case_custom = repo_root / "test_cases" / f"{case_prefix}_incompressibleVoF"
    test_case_native = repo_root / "test_cases" / f"{case_prefix}_{native_solver}"

    try:
        # ------------------------------------------------------------------ #
        # Setup both cases (blockMesh + setFields)                            #
        # ------------------------------------------------------------------ #
        print("\n=== Setting up test cases ===")
        for test_case in (test_case_custom, test_case_native):
            setup_case(
                source_case,
                test_case,
                end_time,
                write_interval,
                run_setfields=True,
            )

        # ------------------------------------------------------------------ #
        # Run incompressibleVoF (advection scheme auto-detected from the      #
        # case's fvSolution)                                                  #
        # ------------------------------------------------------------------ #
        print("\n=== Running incompressibleVoF ===")
        original_dir = Path.cwd()
        os.chdir(test_case_custom)
        try:
            run(["incompressibleVoF"])
        finally:
            os.chdir(original_dir)

        # ------------------------------------------------------------------ #
        # Run the native solver                                               #
        # ------------------------------------------------------------------ #
        print(f"\n=== Running native {native_solver} ===")
        result = subprocess.run(
            [native_solver, "-case", str(test_case_native)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, f"{native_solver} failed:\n{result.stderr}"

        # ------------------------------------------------------------------ #
        # Compare fields (same schemes, same adaptive-dt sequence)            #
        # ------------------------------------------------------------------ #
        all_match, failed_fields, failed_details = compare_solver_fields(
            test_case_custom,
            test_case_native,
            FIELDS_TO_COMPARE,
            rtol=1e-10,
            atol=1e-10,
        )

        if not all_match:
            detail_parts = []
            for field_name in failed_fields:
                max_abs_diff, max_rel_diff = failed_details.get(
                    field_name, (float("nan"), float("nan"))
                )
                detail_parts.append(f"{field_name}(abs={max_abs_diff:.3e}, rel={max_rel_diff:.3e})")
            assert False, "Field values differ beyond tolerance. Failed fields: " + ", ".join(
                detail_parts
            )

        print("\n=== Test PASSED: Results match within tolerance ===")

    finally:
        for test_case in (test_case_custom, test_case_native):
            if test_case.exists():
                shutil.rmtree(test_case)
                print(f"Cleaned up: {test_case}")


def write_dict_entries(
    dict_path: Path, sub_dict: str, entries: Mapping[str, Union[bool, float]]
) -> None:
    """Overwrite entries of one sub-dict of an OpenFOAM dictionary in place.

    Goes through pybFoam's own ``dictionary.read``/``set``/``write``, so the
    test never patches dictionary text (TEST_STYLE rule 3).
    """
    parsed = dictionary.read(str(dict_path))
    sub = parsed.subDict(sub_dict)
    for key, value in entries.items():
        sub.set(key, value)
    parsed.write(str(dict_path))


def write_alpha_controls(
    fv_solution_path: Path, alpha_controls: Mapping[str, Union[bool, float]]
) -> None:
    """Overwrite entries of the ``"alpha.water.*"`` solver dict in place.

    Goes through pybFoam's own ``dictionary.read``/``set``/``write`` — the
    OpenFOAM format's reader/writer — so the test never patches dictionary
    text (TEST_STYLE rule 3). ``subDict("alpha.water")`` resolves the real
    tutorial's regex key ``"alpha.water.*"`` and hands back a reference, so the
    mutations land in the parsed file that is written back out.
    """
    fv_solution = dictionary.read(str(fv_solution_path))
    alpha_dict = fv_solution.subDict("solvers").subDict("alpha.water")
    for key, value in alpha_controls.items():
        alpha_dict.set(key, value)
    fv_solution.write(str(fv_solution_path))


def run_dambreak_regime(
    alpha_controls: Mapping[str, Union[bool, float]],
    python_case: Path,
    native_case: Path,
    end_time: float = 0.05,
    write_interval: float = 0.05,
    fv_schemes: Optional[Path] = None,
    pimple_controls: Optional[Mapping[str, Union[bool, float]]] = None,
) -> None:
    """Set up ``tutorials/damBreak`` twice with ``alpha_controls`` applied, run both.

    ``python_case`` is run with incompressibleVoF **in a subprocess** (one
    ``Foam::Time`` per process — several regimes run in one pytest session),
    ``native_case`` with the native ``interFoam`` binary. Both directories are
    left on disk for the caller to compare.

    ``fv_schemes`` replaces the tutorial's ``system/fvSchemes`` wholesale with a
    checked-in variant — a scheme spec such as ``CrankNicolson 0.5`` is two
    tokens, which the dictionary writer has no way to set as one entry —
    and ``pimple_controls`` overwrites entries of the ``PIMPLE`` dict.
    """
    for case in (python_case, native_case):
        setup_case(
            _REPO_ROOT / "tutorials" / "damBreak",
            case,
            end_time,
            write_interval,
            run_setfields=True,
        )
        if fv_schemes is not None:
            shutil.copyfile(fv_schemes, case / "system" / "fvSchemes")
        write_alpha_controls(case / "system" / "fvSolution", alpha_controls)
        if pimple_controls:
            write_dict_entries(case / "system" / "fvSolution", "PIMPLE", pimple_controls)

    custom = subprocess.run(
        [sys.executable, str(_SOLVER_WORKER), str(python_case)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert custom.returncode == 0, (
        f"incompressibleVoF failed for {dict(alpha_controls)}:\n"
        f"{custom.stdout[-4000:]}\n{custom.stderr[-4000:]}"
    )

    native = subprocess.run(
        ["interFoam", "-case", str(native_case)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert native.returncode == 0, (
        f"interFoam failed for {dict(alpha_controls)}:\n{native.stderr[-4000:]}"
    )
