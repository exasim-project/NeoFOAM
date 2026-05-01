# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Base interface for NeoN turbulence models with PluginSystem dispatch."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from neofoam.core.plugin_system import PluginSystem
from neofoam.framework.context import Context, FieldUpdates


@PluginSystem.register(discriminator_variable="config", discriminator="turbulence_type")
class NeonTurbulenceModel(BaseModel):
    """Base class for NeoN turbulence models.

    Subclasses register via @NeonTurbulenceModel.register and must implement:
    - build_steps(): return InitStep list for fields/models
    - correct(ctx): solve turbulence equations and update fields

    Each subclass must also define:
    - turbulence_type: Literal["..."] discriminator
    - detect_model(): static method returning bool
    """

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def detect(cls) -> NeonTurbulenceModel:
        """Detect turbulence model from turbulenceProperties.

        Iterates registered plugins, calls detect_model() on each.
        Falls back to laminar if none match.
        """
        registry = PluginSystem.get_registered("NeonTurbulenceModel")
        if registry:
            for plugin_cls in registry.plugin_registry:
                if hasattr(plugin_cls, "detect_model") and plugin_cls.detect_model():
                    return plugin_cls()
        from neofoam.turbulenceModels.laminar import NeonLaminar

        return NeonLaminar()

    @classmethod
    def registered_models(cls) -> list[str]:
        """Return names of all registered turbulence model types."""
        registry = PluginSystem.get_registered("NeonTurbulenceModel")
        if registry:
            return registry.get_plugin_names()
        return []

    def build_steps(self) -> list[Any]:
        """Return InitStep objects for fields and models this model needs."""
        raise NotImplementedError

    def correct(self, ctx: Context) -> FieldUpdates:
        """Correct turbulence fields. Returns updated fields."""
        raise NotImplementedError
