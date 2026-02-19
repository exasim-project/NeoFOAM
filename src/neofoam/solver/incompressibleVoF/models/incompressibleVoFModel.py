# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""Plugin interface and helpers for incompressibleVoF solver models."""

from typing import Any

from pydantic import BaseModel

from neofoam.core.plugin_system import PluginSystem
from neofoam.framework.model_factory import ModelInstance


def Model(name: str) -> ModelInstance:
    """Factory for incompressibleVoF models using ModelInstance API."""
    return ModelInstance(name)


@PluginSystem.register(discriminator_variable="model", discriminator="model_type")
class incompressibleVoFModel(BaseModel):
    """Plugin interface for solver-local incompressibleVoF models."""

    @classmethod
    def detect_models(cls) -> list[ModelInstance]:
        """Return enabled model instances from plugin registry."""
        registry = PluginSystem.get_registered("incompressibleVoFModel")
        if not registry:
            return []

        enabled_models: list[ModelInstance] = []
        for plugin_cls in registry.plugin_registry:
            if not hasattr(plugin_cls, "get_model_instance"):
                continue

            model_instance = plugin_cls.get_model_instance(plugin_cls)
            if model_instance.run_detect():
                enabled_models.append(model_instance)

        return enabled_models


def _create_model_instance(cls, *, config: dict[str, Any], **kwargs: Any) -> Any:
    """Create registered model instance from discriminator config."""
    wrapper = cls.plugin_model(model=config, **kwargs)  # type: ignore[attr-defined]
    return wrapper.model.get_model_instance()


incompressibleVoFModel.create = classmethod(_create_model_instance)  # type: ignore[method-assign]
