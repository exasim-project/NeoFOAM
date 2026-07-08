# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Plugin interface for solver-local incompressibleFluidNeoN optional models.

Mirror of the pybFoam solver's ``incompressibleFluidModel`` interface for the
NeoN-backed solver. The PluginSystem registry is keyed by this class name, so
the NeoN family is disjoint from the pybFoam one — a model registered here is
only discovered by ``incompressibleFluidNeoN``.
"""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from neofoam.core.plugin_system import PluginSystem
from neofoam.framework.model import Model, ModelRuntime, ModelSpec  # noqa: F401

__all__ = ["incompressibleFluidNeoNModel", "Model", "ModelRuntime", "ModelSpec"]


@PluginSystem.register(discriminator_variable="model", discriminator="model_type")
class incompressibleFluidNeoNModel(BaseModel):
    """Plugin interface for solver-local incompressibleFluidNeoN models."""

    @classmethod
    def all_specs(cls) -> list[ModelSpec]:
        """Return every registered optional-model spec, without detection.

        Unlike :meth:`detect_models`, this does not run ``detect`` (which
        needs a case) — it lists the full set of optional models so their
        config classes can be enumerated case-free.
        """
        registry = PluginSystem.get_registered("incompressibleFluidNeoNModel")
        if not registry:
            return []

        return [
            plugin_cls.get_model_instance(plugin_cls)
            for plugin_cls in registry.plugin_registry
            if hasattr(plugin_cls, "get_model_instance")
        ]

    @classmethod
    def detect_models(cls, case_dir: Optional[Path] = None) -> list[ModelRuntime]:
        """Return enabled model runtimes from the plugin registry."""
        registry = PluginSystem.get_registered("incompressibleFluidNeoNModel")
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
