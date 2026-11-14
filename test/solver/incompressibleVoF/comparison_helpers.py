# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Shared damBreak comparison harness: incompressibleVoF vs a native solver.

Both damBreak comparisons (MULES vs interFoam, isoAdvector vs interIsoFoam) run
the identical sequence — set up two copies of a damBreak tutorial, run the
Python solver in one and the native OpenFOAM solver in the other, then compare
the written fields — so the whole body lives here and
``test_damBreak_comparison.py`` is one parametrized test over the two.
A :class:`SubjectSolver` swaps in another Python solver (run in a subprocess,
compared by relative L2 norm per write time): ``incompressibleVoFNeoN`` uses it.

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
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Union

import numpy as np
from pybFoam import dictionary

from neofoam.io import DictFile
from neofoam.solver.incompressibleVoF import run

# Re-use the comparison helpers from the sibling incompressibleFluid test
# package (``test/solver`` is a package, so a ``..`` relative import reaches it).
from ..incompressibleFluid.comparison_helpers import (
    compare_solver_fields,
    get_time_directories,
    read_internal_fields,
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


@dataclass(frozen=True)
class SubjectSolver:
    """A Python solver compared against the native one by relative L2 norm.

    ``command`` runs from the case directory in its own process (one
    ``Foam::Time`` and Kokkos runtime each); ``prepare_case`` edits each staged
    case before either solver runs; ``rel_l2_ceilings`` maps a compared field
    name to the largest relative L2 difference ``|subject - native|/|native|``
    accepted at any write time.
    """

    name: str
    command: list[str]
    prepare_case: Callable[[Path], None]
    rel_l2_ceilings: Mapping[str, float]


def _run_subject(subject: SubjectSolver, case: Path) -> None:
    """Run the subject solver in ``case``, its log in ``<case>/log.<name>``."""
    log_path = case / f"log.{subject.name}"
    with log_path.open("w") as log:
        completed = subprocess.run(
            subject.command, cwd=case, stdout=log, stderr=subprocess.STDOUT, timeout=1200
        )
    assert completed.returncode == 0, (
        f"{subject.name} failed in {case}:\n{log_path.read_text()[-4000:]}"
    )


def _rel_l2(subject: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(subject - reference) / (np.linalg.norm(reference) + 1e-300))


def _assert_rel_l2(
    subject_case: Path, native_case: Path, subject: SubjectSolver
) -> dict[float, dict[str, float]]:
    """relL2 of every compared field at every write time both cases wrote (time 0 excluded)."""
    native_times = {float(d.name): d for d in get_time_directories(native_case)}
    common = [d for d in get_time_directories(subject_case) if float(d.name) in native_times]
    common = [d for d in common if float(d.name) > 0]
    assert common, f"{subject.name} and the native solver share no write time"

    names = list(subject.rel_l2_ceilings)
    measured: dict[float, dict[str, float]] = {}
    for time_dir in common:
        time = float(time_dir.name)
        ours = read_internal_fields(subject_case, time_dir, names)
        theirs = read_internal_fields(native_case, native_times[time], names)
        measured[time] = {name: _rel_l2(ours[name], theirs[name]) for name in names}
        print(f"t = {time}: " + ", ".join(f"{n} {v:.3e}" for n, v in measured[time].items()))

    exceeded = [
        f"{name} at t={time}: {values[name]:.3e} > {subject.rel_l2_ceilings[name]:.1e}"
        for time, values in measured.items()
        for name in names
        if values[name] > subject.rel_l2_ceilings[name]
    ]
    assert not exceeded, f"{subject.name} vs native relL2 above ceiling: " + "; ".join(exceeded)
    return measured


def run_dambreak_comparison(
    tutorial_name: str,
    native_solver: str,
    case_prefix: str,
    times: tuple[float, float] = (0.05, 0.05),
    subject: Optional[SubjectSolver] = None,
) -> dict[float, dict[str, float]]:
    """Run ``tutorials/<tutorial_name>`` with a Python solver and with the
    native ``native_solver`` binary, then compare the written fields.

    Without ``subject`` the Python solver is incompressibleVoF, run in this
    process, and the fields at the final time must match to machine precision
    (rtol=atol=1e-10); the result is then empty. With ``subject`` its
    ``command`` runs in a subprocess from the case directory, its
    ``prepare_case`` edits both cases first, and every common write time is
    compared by relative L2 norm against ``rel_l2_ceilings``; the measured
    ``{time: {field: relL2}}`` is returned.

    ``times`` is ``(end_time, write_interval)``. ``case_prefix`` names the two
    scratch case directories under ``test_cases/`` (``<case_prefix>_<solver>`` /
    ``<case_prefix>_<native_solver>``), which are removed afterwards.
    """
    end_time, write_interval = times
    repo_root = Path(__file__).parent.parent.parent.parent
    source_case = repo_root / "tutorials" / tutorial_name

    subject_name = "incompressibleVoF" if subject is None else subject.name
    test_case_custom = repo_root / "test_cases" / f"{case_prefix}_{subject_name}"
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
            if subject is not None:
                subject.prepare_case(test_case)

        # ------------------------------------------------------------------ #
        # Run incompressibleVoF (advection scheme auto-detected from the      #
        # case's fvSolution)                                                  #
        # ------------------------------------------------------------------ #
        print(f"\n=== Running {subject_name} ===")
        if subject is None:
            original_dir = Path.cwd()
            os.chdir(test_case_custom)
            try:
                run(["incompressibleVoF"])
            finally:
                os.chdir(original_dir)
        else:
            _run_subject(subject, test_case_custom)

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

        if subject is not None:
            return _assert_rel_l2(test_case_custom, test_case_native, subject)

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
        return {}

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


def write_ddt_schemes(fv_schemes_path: Path, entries: Mapping[str, str]) -> None:
    """Overwrite entries of ``system/fvSchemes``' ``ddtSchemes`` dict in place.

    Through :class:`neofoam.io.DictFile` — the OpenFOAM format's own
    reader/writer — so the test never patches dictionary text (TEST_STYLE
    rule 3). A scheme spec is two tokens (``CrankNicolson 0.5``); the writer
    emits the value verbatim, so it needs no checked-in fvSchemes variant.
    """
    dict_file = DictFile(fv_schemes_path)
    for key, value in entries.items():
        dict_file.set(("ddtSchemes", key), value)
    dict_file.write()


def run_dambreak_regime(
    alpha_controls: Mapping[str, Union[bool, float]],
    python_case: Path,
    native_case: Path,
    end_time: float = 0.05,
    write_interval: float = 0.05,
    ddt_schemes: Optional[Mapping[str, str]] = None,
    pimple_controls: Optional[Mapping[str, Union[bool, float]]] = None,
) -> None:
    """Set up ``tutorials/damBreak`` twice with ``alpha_controls`` applied, run both.

    ``python_case`` is run with incompressibleVoF **in a subprocess** (one
    ``Foam::Time`` per process — several regimes run in one pytest session),
    ``native_case`` with the native ``interFoam`` binary. Both directories are
    left on disk for the caller to compare.

    ``ddt_schemes`` overwrites entries of the tutorial's ``ddtSchemes`` dict and
    ``pimple_controls`` entries of its ``PIMPLE`` dict.
    """
    for case in (python_case, native_case):
        setup_case(
            _REPO_ROOT / "tutorials" / "damBreak",
            case,
            end_time,
            write_interval,
            run_setfields=True,
        )
        if ddt_schemes:
            write_ddt_schemes(case / "system" / "fvSchemes", ddt_schemes)
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
