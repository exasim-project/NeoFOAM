# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""Plugin interface and helpers for incompressibleFluid solver models."""

from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel

from neofoam.core.plugin_system import PluginSystem
from neofoam.framework.model import Model, ModelRuntime, ModelSpec  # noqa: F401


@PluginSystem.register(discriminator_variable="model", discriminator="model_type")
class incompressibleFluidModel(BaseModel):
    """Plugin interface for solver-local incompressibleFluid models."""

    @classmethod
    def detect_models(cls, case_dir: Optional[Path] = None) -> list[ModelRuntime]:
        """Return enabled model runtimes from plugin registry."""
        registry = PluginSystem.get_registered("incompressibleFluidModel")
        if not registry:
            return []

        runtimes: list[ModelRuntime] = []
        effective_dir = case_dir or Path(".")
        for plugin_cls in registry.plugin_registry:
            if not hasattr(plugin_cls, "get_model_instance"):
                continue

            spec: ModelSpec = plugin_cls.get_model_instance(plugin_cls)
            detect_result = spec.run_detect(case_dir=effective_dir)
            if detect_result.detected:
                entry: dict[str, Any] = {"type": spec.name, "name": spec.name}
                runtimes.append(spec.instantiate(case_dir=effective_dir, entry=entry))

        return runtimes


def _create_model_instance(cls, *, config: dict[str, Any], **kwargs: Any) -> Any:
    """Create registered model instance from discriminator config."""
    wrapper = cls.plugin_model(model=config, **kwargs)  # type: ignore[attr-defined]
    return wrapper.model.get_model_instance()


incompressibleFluidModel.create = classmethod(_create_model_instance)  # type: ignore[method-assign]
