# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Base class for DummySolver models.

Mimics SimpleSolverModel structure for testing.
"""

from pydantic import BaseModel

from neofoam.core.plugin_system import PluginSystem
from neofoam.framework.model import ModelSpec


def Model(name: str) -> ModelSpec:
    """Factory for dummy models using the ModelSpec API."""
    return ModelSpec(name)


@PluginSystem.register(discriminator_variable="model", discriminator="model_type")
class DummyModelInterface(BaseModel):
    """
    Base class for DummySolver optional models.

    Provides infrastructure for automatic model detection and integration.
    """

    @classmethod
    def detect_specs(cls) -> list[ModelSpec]:
        """
        Return all registered ModelSpec objects that pass their detect() check.

        Callers call ``spec.instantiate(case_dir, instance_id)`` per config entry.
        """
        registry = PluginSystem.get_registered("DummyModelInterface")
        if not registry:
            return []

        return [
            spec
            for plugin_cls in registry.plugin_registry
            if hasattr(plugin_cls, "get_model_instance")
            for spec in [plugin_cls.get_model_instance(plugin_cls)]
            if spec.run_detect()
        ]
