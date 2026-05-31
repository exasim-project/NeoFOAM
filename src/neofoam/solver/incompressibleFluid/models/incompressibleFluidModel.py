# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Plugin interface for solver-local incompressibleFluid optional models.

The PluginSystem registration keeps the extension point identical to the
source branch so future optional models (boussinesq, spalartAllmaras, …)
can be re-added without changing the solver entrypoint.
"""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from neofoam.core.plugin_system import PluginSystem
from neofoam.framework.model import Model, ModelRuntime, ModelSpec  # noqa: F401

__all__ = ["incompressibleFluidModel", "Model", "ModelRuntime", "ModelSpec"]


@PluginSystem.register(discriminator_variable="model", discriminator="model_type")
class incompressibleFluidModel(BaseModel):
    """Plugin interface for solver-local incompressibleFluid models."""

    @classmethod
    def all_specs(cls) -> list[ModelSpec]:
        """Return every registered optional-model spec, without detection.

        Unlike :meth:`detect_models`, this does not run ``detect`` (which
        needs a case) — it lists the full set of optional models so their
        config classes can be enumerated case-free.
        """
        registry = PluginSystem.get_registered("incompressibleFluidModel")
        if not registry:
            return []

        return [
            plugin_cls.get_model_instance(plugin_cls)
            for plugin_cls in registry.plugin_registry
            if hasattr(plugin_cls, "get_model_instance")
        ]

    @classmethod
    def detect_models(cls, case_dir: Optional[Path] = None) -> list[ModelRuntime]:
        """Return enabled model runtimes from the plugin registry.

        Empty by default in this minimal port; any optional model that
        registers itself with this interface will be discovered here.
        """
        registry = PluginSystem.get_registered("incompressibleFluidModel")
        if not registry:
            return []

        runtimes: list[ModelRuntime] = []
        effective_dir = case_dir or Path(".")
        for plugin_cls in registry.plugin_registry:
            if not hasattr(plugin_cls, "get_model_instance"):
                continue

            spec: ModelSpec = plugin_cls.get_model_instance(plugin_cls)
            if spec.run_detect():
                runtimes.append(
                    spec.instantiate(case_dir=effective_dir, instance_id=spec.name)
                )

        return runtimes
