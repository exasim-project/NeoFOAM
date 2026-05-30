# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Viscosity plugin interface.

A viscosity model supplies the molecular kinematic viscosity ``nu``. It is a
core model family of the incompressible solver: ``laminarTransport`` (the pybFoam
``singlePhaseTransportModel``) and a native viscosity model are the *same
abstraction* — the molecular-``nu`` provider, OpenFOAM vs. native.

Native models register here via ``Model("name").register_with(viscosityModel)``
and are discovered case-free by name. There is **no per-model class and no
adapter**: a model is a :class:`ModelSpec` + config + operations and **owns its
``nu`` field on the Context**, registering/updating it through its operations.
The viscosity model deliberately has no read interface — its only consumer is the
momentum-transport model, which reads ``nu`` as a Context field.
"""

from typing import Any, Optional

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

    @classmethod
    def detect_and_create(cls) -> Any:
        """Select the active model for the case in the current directory.

        The core-model-family hook: reads ``constant/transportProperties`` and
        returns the matching native spec or the OpenFOAM fallback adapter.
        """
        from .selection import select_from_case

        return select_from_case()
