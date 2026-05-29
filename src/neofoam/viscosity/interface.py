# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Plugin interface for viscosity (transport) models.

Standalone viscosity subsystem mirroring :mod:`neofoam.turbulence`: transport
models register themselves via ``Model("name").register_with(viscosityModel)``
and are discovered case-free by name. ``constant/transportProperties`` is a
case-global dictionary, so the subsystem is not tied to a single solver.

This module is intentionally free of any ``neofoam.io`` / pybFoam import so the
registry and selection logic stay importable without a built OpenFOAM
environment.
"""

from typing import Optional

from pydantic import BaseModel

from neofoam.core.plugin_system import PluginSystem
from neofoam.framework.model import Model, ModelRuntime, ModelSpec  # noqa: F401

__all__ = ["viscosityModel", "Model", "ModelRuntime", "ModelSpec"]


@PluginSystem.register(discriminator_variable="model", discriminator="model_type")
class viscosityModel(BaseModel):
    """Plugin interface for viscosity (transport) models."""

    @classmethod
    def all_specs(cls) -> list[ModelSpec]:
        """Return every registered viscosity spec, without detection."""
        registry = PluginSystem.get_registered("viscosityModel")
        if not registry:
            return []

        return [
            plugin_cls.get_model_instance(plugin_cls)
            for plugin_cls in registry.plugin_registry
            if hasattr(plugin_cls, "get_model_instance")
        ]

    @classmethod
    def registered_names(cls) -> list[str]:
        """Return the ``spec.name`` of every registered viscosity model."""
        return [spec.name for spec in cls.all_specs()]

    @classmethod
    def find_spec(cls, name: str) -> Optional[ModelSpec]:
        """Return the registered spec whose ``name`` matches, else ``None``."""
        for spec in cls.all_specs():
            if spec.name == name:
                return spec
        return None
