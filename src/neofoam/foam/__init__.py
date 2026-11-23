# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""OpenFOAM-specific utilities and per-model dictionary base classes.

Currently exports:

- ``fvSchemes`` / ``fvSolution`` — ``BaseConfig`` base classes
  bound to ``system/fvSchemes`` and ``system/fvSolution``. Register
  via ``spec.config(fvSchemes)`` to receive a per-spec subclass that
  operations extend via ``@<Subclass>.add(...)``.
- ``neonFvSolution`` / ``openfoamFvSolution`` / ``LinearSolver`` — the typed ``fvSolution``
  slices and their linear-solver plugin family; each solver's form lists one backend's plugins.
- ``schemes`` — typed scheme unions (DdtScheme, DivScheme, …) used as
  the value types when ``.add(...)`` injects fields into a subclass.
- ``algorithm_configs`` — typed views of ``fvSolution``'s algorithm-control
  blocks (``PIMPLE`` / ``PISO`` / ``SIMPLE``), which the per-spec slices above
  pass through untyped.
"""

from neofoam.core.field_metadata import advanced
from neofoam.foam.algorithm_configs import (
    DynamicMeshControls,
    PimpleAlgorithmConfig,
    PisoAlgorithmConfig,
    PisoDynamicMeshControls,
    SimpleAlgorithmConfig,
)
from neofoam.foam.fv_configs import fvSchemes, fvSolution
from neofoam.foam.typed_fv_solution import neonFvSolution, openfoamFvSolution
from neofoam.linear_solvers import (
    GinkgoConfigFile,
    GinkgoPreconditioner,
    GinkgoSolver,
    LinearSolver,
    LinearSolverConversionWarning,
    Preconditioner,
    linear_solver_entry,
    refresh_linear_solver_slices,
)

__all__ = [
    "DynamicMeshControls",
    "GinkgoConfigFile",
    "GinkgoPreconditioner",
    "GinkgoSolver",
    "LinearSolver",
    "LinearSolverConversionWarning",
    "PimpleAlgorithmConfig",
    "PisoAlgorithmConfig",
    "PisoDynamicMeshControls",
    "Preconditioner",
    "SimpleAlgorithmConfig",
    "advanced",
    "fvSchemes",
    "fvSolution",
    "linear_solver_entry",
    "neonFvSolution",
    "openfoamFvSolution",
    "refresh_linear_solver_slices",
]
