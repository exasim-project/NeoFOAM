# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

# neofoam package
__version__ = "0.0.1"

from importlib import import_module

try:
    # First: the bindings' MeshAdapter declares Foam::fvMesh as its base, and pybFoam
    # is what registers that type in the shared nanobind registry.
    import pybFoam as pybFoam  # noqa: F401

    # Not `from . import`: that reports a module that is not built as a plain ImportError.
    neofoam_bindings = import_module("neofoam.neofoam_bindings")
# Only a stack that is not built: a broken one (nanobind ABI, missing shared library)
# is a plain ImportError and has to show its cause.
except ModuleNotFoundError:
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
