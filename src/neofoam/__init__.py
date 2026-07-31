# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

# neofoam package
__version__ = "0.0.1"

try:
    # pybFoam registers the OpenFOAM types (Foam::fvMesh, …) in the nanobind type
    # registry both packages share; the bindings' MeshAdapter declares fvMesh as its
    # base, so pybFoam has to be imported first.
    import pybFoam as pybFoam  # noqa: F401

    from . import neofoam_bindings as neofoam_bindings  # type: ignore[attr-defined,unused-ignore]
except ImportError:
    neofoam_bindings = None  # type: ignore[assignment,unused-ignore]

from .framework.context import FieldUpdates
from .framework.initialization import Depends, field
from .framework.model import Model
from .framework.solver import Configurations, Solver, configurations

__all__ = [
    "Configurations",
    "Depends",
    "FieldUpdates",
    "Model",
    "Solver",
    "configurations",
    "field",
]
