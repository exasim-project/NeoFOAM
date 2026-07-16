# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""
Comparison test for damBreak: incompressibleVoF (isoAdvector) vs native interIsoFoam.

The ``damBreak_isoAdvector`` tutorial selects the geometric advection scheme via
``advectionScheme isoAdvector;`` in ``system/fvSolution`` (with the isoAdvector
controls in the ``"alpha.water.*"`` solver sub-dict). isoAdvector is a pure
function of ``(alpha1, phi, U, deltaT)``; both solvers read identical controls
and share the same adaptive-dt routines (``pybFoam.computeCFLNumber`` and the
Python-composed ``compute_alpha_courant_number``), so they follow the same
time-step sequence deterministically and results must match to machine
precision (rtol=1e-10).

Companion to ``test_damBreak_comparison.py`` (the MULES scheme vs interFoam);
the shared body lives in ``comparison_helpers``.
"""

from .comparison_helpers import run_dambreak_comparison


def test_damBreak_isoAdvector_solver_comparison():
    """Compare incompressibleVoF (isoAdvector) against native interIsoFoam."""
    run_dambreak_comparison(
        tutorial_name="damBreak_isoAdvector",
        native_solver="interIsoFoam",
        case_prefix="damBreak_iso",
    )
