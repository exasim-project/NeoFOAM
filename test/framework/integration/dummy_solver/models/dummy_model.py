# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Base class for DummySolver models.

Mimics SimpleSolverModel structure for testing.
"""

from typing import Any

from pydantic import BaseModel

from neofoam.core.plugin_system import PluginSystem
from neofoam.framework.model_factory import ModelInstance


def Model(name: str) -> ModelInstance:
    """
    Factory for dummy models using the ModelInstance API.
    """
    return ModelInstance(name)


@PluginSystem.register(discriminator_variable="model", discriminator="model_type")
class DummyModelInterface(BaseModel):
    """
    Base class for DummySolver optional models.

    Provides infrastructure for automatic model detection and integration.
    """

    @classmethod
    def create(cls, *, config: dict[str, Any], **kwargs: Any) -> Any:
        """Factory classmethod for creating model instances."""
        wrapper = cls.plugin_model(model=config, **kwargs)  # type: ignore[attr-defined]
        return wrapper.model.get_model_instance()

    @classmethod
    def detect_models(cls) -> list[ModelInstance]:
        """
        Detect and return enabled model instances.

        Returns:
            List of ModelInstance objects that have been registered and enabled
        """
        registry = PluginSystem.get_registered("DummyModelInterface")
        if not registry:
            return []

        enabled_models = []
        for plugin_cls in registry.plugin_registry:
            # Get the ModelInstance from the wrapper class
            if hasattr(plugin_cls, "get_model_instance"):
                model_instance = plugin_cls.get_model_instance(plugin_cls)
                # Check if model should be detected/enabled
                if model_instance.run_detect():
                    enabled_models.append(model_instance)

        return enabled_models
