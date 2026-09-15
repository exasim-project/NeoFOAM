# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""OpenFOAM-specific utilities and per-model dictionary base classes.

Currently exports:

- ``fvSchemes`` / ``fvSolution`` — ``BaseConfig`` base classes
  bound to ``system/fvSchemes`` and ``system/fvSolution``. Register
  via ``spec.config(fvSchemes)`` to receive a per-spec subclass that
  operations extend via ``@<Subclass>.add(...)``.
- ``schemes`` — typed scheme unions (DdtScheme, DivScheme, …) used as
  the value types when ``.add(...)`` injects fields into a subclass.
- ``algorithm_configs`` — typed views of ``fvSolution``'s algorithm-control
  blocks (``PIMPLE`` / ``PISO`` / ``SIMPLE``), which the per-spec slices above
  pass through untyped.
"""

from neofoam.foam.algorithm_configs import (
    DynamicMeshControls,
    PimpleAlgorithmConfig,
    PisoAlgorithmConfig,
    PisoDynamicMeshControls,
    SimpleAlgorithmConfig,
)
from neofoam.foam.fv_configs import fvSchemes, fvSolution

__all__ = [
    "DynamicMeshControls",
    "PimpleAlgorithmConfig",
    "PisoAlgorithmConfig",
    "PisoDynamicMeshControls",
    "SimpleAlgorithmConfig",
    "fvSchemes",
    "fvSolution",
]
