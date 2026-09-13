# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""incompressibleVoF against the native OpenFOAM solver, one tutorial per scheme.

Both advection schemes are run end to end here, one ``parametrize`` entry each
(TEST_STYLE rule 10) — the two runs share every step of the harness in
``comparison_helpers``:

* ``damBreak`` vs ``interFoam`` — the MULES scheme at the tutorial's own
  controls (``nAlphaCorr 2; nAlphaSubCycles 1; cAlpha 1; MULESCorr yes``). The
  rest of that configuration space is ``test_mules_regimes.py``.
* ``damBreak_isoAdvector`` vs ``interIsoFoam`` — this repo's tutorial, which
  selects the geometric scheme with ``advectionScheme isoAdvector;`` in
  ``system/fvSolution`` (isoAdvector's controls sitting in the
  ``"alpha.water.*"`` solver sub-dict). isoAdvector is a pure function of
  ``(alpha1, phi, U, deltaT)``.

Both solvers use adaptive time stepping (``adjustTimeStep yes, maxCo 1,
maxAlphaCo 1``) and share the CFL routines (``pybFoam.computeCFLNumber`` and the
Python-composed ``compute_alpha_courant_number``), so they follow the same
time-step sequence deterministically and the fields must match to machine
precision (``rtol = atol = 1e-10``), not to a modelling tolerance.
"""

import pytest

from .comparison_helpers import run_dambreak_comparison


@pytest.mark.parametrize(
    "tutorial_name, native_solver, case_prefix",
    [
        pytest.param("damBreak", "interFoam", "damBreak", id="MULES"),
        pytest.param("damBreak_isoAdvector", "interIsoFoam", "damBreak_iso", id="isoAdvector"),
    ],
)
def test_damBreak_matches_the_native_solver(
    tutorial_name: str, native_solver: str, case_prefix: str
) -> None:
    run_dambreak_comparison(
        tutorial_name=tutorial_name,
        native_solver=native_solver,
        case_prefix=case_prefix,
    )
