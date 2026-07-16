# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Comparison test for damBreak: incompressibleVoF vs native interFoam.

Both solvers use adaptive time stepping (adjustTimeStep yes, maxCo 1, maxAlphaCo 1).
Since both use identical CFL computation (pybFoam.computeCFLNumber and the
Python-composed compute_alpha_courant_number, built from the same OpenFOAM
primitives), they follow the same time-step sequence deterministically and
results must match to machine precision (rtol=1e-10).

Companion to ``test_damBreak_isoAdvector_comparison.py`` (the isoAdvector
scheme vs interIsoFoam); the shared body lives in ``comparison_helpers``.
"""

from .comparison_helpers import run_dambreak_comparison


def test_damBreak_solver_comparison():
    """Compare incompressibleVoF against native interFoam on damBreak case."""
    run_dambreak_comparison(
        tutorial_name="damBreak",
        native_solver="interFoam",
        case_prefix="damBreak",
    )
