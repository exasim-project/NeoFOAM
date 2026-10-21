# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The regex-keyed case, composed as a one-file delta over ``test/setup_pimple``.

Only ``system/fvSolution`` differs from the laminar lid-driven cavity the other
runs here use, so only that file is committed (``cases/regexSolverKeys/``) and it
is laid over the cavity — the layout of ``test/turbulence/_parity_case.py``. It
has to be a tree overlay, not a ``patch`` pipeline: pybFoam's ``dictionary.set``
strips the quotes off a keyword, so the regex-keyed *scalar*
``"(U|k|epsilon)" 0.7;`` under ``relaxationFactors.equations`` cannot be written
at all (see ``test/tooling/casebuild/test_steps.py``).
"""

from __future__ import annotations

import shutil
from pathlib import Path

from neofoam.tooling.casebuild import CaseDir, Pipeline, from_template

_HERE = Path(__file__).parent

#: The complete case the delta is laid over.
SETUP_PIMPLE = _HERE.parents[1] / "setup_pimple"

#: Everything the regex case changes: ``system/fvSolution``. Not runnable alone.
REGEX_DELTA = _HERE / "cases" / "regexSolverKeys"


def regex_solver_keys_case() -> Pipeline:
    """``test/setup_pimple`` with the regex-keyed ``system/fvSolution`` laid over it."""

    def overlay(case: CaseDir) -> None:
        shutil.copytree(REGEX_DELTA, case.path, dirs_exist_ok=True)

    return from_template(SETUP_PIMPLE) | overlay
