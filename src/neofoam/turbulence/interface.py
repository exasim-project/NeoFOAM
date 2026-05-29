# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Plugin interface for turbulence models.

Standalone turbulence subsystem: turbulence models register themselves with
this interface via ``Model("name").register_with(turbulenceModel)`` and are
discovered case-free by name. Mirrors
``solver/incompressibleFluid/models/incompressibleFluidModel.py`` but is not
tied to a single solver, since ``constant/turbulenceProperties`` is a
case-global dictionary.

This module is intentionally free of any ``neofoam.io`` / pybFoam import so the
registry and selection logic stay importable without a built OpenFOAM
environment.
"""

from typing import Optional

from pydantic import BaseModel

from neofoam.core.plugin_system import PluginSystem
from neofoam.framework.model import Model, ModelRuntime, ModelSpec  # noqa: F401

__all__ = ["turbulenceModel", "Model", "ModelRuntime", "ModelSpec"]


@PluginSystem.register(discriminator_variable="model", discriminator="model_type")
class turbulenceModel(BaseModel):
    """Plugin interface for turbulence models."""

    @classmethod
    def all_specs(cls) -> list[ModelSpec]:
        """Return every registered turbulence spec, without detection.

        Lists the full set of registered models so their names can be matched
        case-free; unlike detection this does not need a concrete case.
        """
        registry = PluginSystem.get_registered("turbulenceModel")
        if not registry:
            return []

        return [
            plugin_cls.get_model_instance(plugin_cls)
            for plugin_cls in registry.plugin_registry
            if hasattr(plugin_cls, "get_model_instance")
        ]

    @classmethod
    def registered_names(cls) -> list[str]:
        """Return the ``spec.name`` of every registered turbulence model."""
        return [spec.name for spec in cls.all_specs()]

    @classmethod
    def find_spec(cls, name: str) -> Optional[ModelSpec]:
        """Return the registered spec whose ``name`` matches, else ``None``."""
        for spec in cls.all_specs():
            if spec.name == name:
                return spec
        return None
