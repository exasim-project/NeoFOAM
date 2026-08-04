# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
neofoam.framework.model — ModelSpec / ModelRuntime package.

Public API:
    ModelSpec    — immutable model definition (replaces ModelInstance)
    ModelRuntime — one instantiation of a spec, owns per-instance config
    Model        — factory alias: Model("Name") -> ModelSpec
"""

from .extension import (
    BoundExtension,
    Extension,
    Hook,
    fold,
    negated,
)
from .runtime import ModelRuntime
from .spec import Model, ModelSpec

__all__ = [
    "ModelSpec",
    "ModelRuntime",
    "Model",
    "BoundExtension",
    "Extension",
    "Hook",
    "fold",
    "negated",
]
