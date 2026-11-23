# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Linear-solver and preconditioner plugins, one module per backend.

Importing this package registers the Ginkgo and OpenFOAM plugins. The binding to
``system/fvSolution`` lives in :mod:`neofoam.foam.typed_fv_solution`.
"""

from neofoam.linear_solvers.base import (
    LinearSolver,
    LinearSolverConversionWarning,
    Preconditioner,
    linear_solver_entry,
    refresh_linear_solver_slices,
    registered_preconditioners,
)
from neofoam.linear_solvers.ginkgo import GinkgoConfigFile, GinkgoPreconditioner, GinkgoSolver
from neofoam.linear_solvers.ginkgo_reader import FOAM_TO_GINKGO_SOLVERS
from neofoam.linear_solvers.openfoam import OpenFoamPreconditioner, OpenFoamSolver

__all__ = [
    "FOAM_TO_GINKGO_SOLVERS",
    "GinkgoConfigFile",
    "GinkgoPreconditioner",
    "GinkgoSolver",
    "LinearSolver",
    "LinearSolverConversionWarning",
    "OpenFoamPreconditioner",
    "OpenFoamSolver",
    "Preconditioner",
    "linear_solver_entry",
    "refresh_linear_solver_slices",
    "registered_preconditioners",
]
