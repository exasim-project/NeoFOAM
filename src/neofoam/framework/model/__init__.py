# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
neofoam.framework.model — ModelSpec / ModelRuntime package.

Public API:
    ModelSpec    — immutable model definition (replaces ModelInstance)
    ModelRuntime — one instantiation of a spec, owns per-instance config
    Model        — factory alias: Model("Name") -> ModelSpec
"""

from .spec import ModelSpec, Model
from .runtime import ModelRuntime
from .interface import (
    BoundModelInterface,
    ModelInterface,
    active_contributors,
    bind_model_interface,
    bind_owned_interfaces,
)

__all__ = [
    "ModelSpec",
    "ModelRuntime",
    "Model",
    "ModelInterface",
    "BoundModelInterface",
    "active_contributors",
    "bind_model_interface",
    "bind_owned_interfaces",
]
