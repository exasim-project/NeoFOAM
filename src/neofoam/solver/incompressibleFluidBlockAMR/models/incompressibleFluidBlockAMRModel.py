# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Plugin interface for solver-local incompressibleFluidBlockAMR optional models.

Mirror of the pybFoam solver's ``incompressibleFluidModel`` interface for the
block-structured AMReX solver. The PluginSystem registry is keyed by this class
name, so the blockAMR family is disjoint from the pybFoam / NeoN ones — a model
registered here is only discovered by ``incompressibleFluidBlockAMR``.

Spec 01 ships no optional models (the laminar box/periodic core needs none), so
``detect_models`` returns an empty list; the interface exists so future
contribution models (e.g. a CFL time-step constraint) register cleanly.
"""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from neofoam.core.plugin_system import PluginSystem
from neofoam.framework.model import Model, ModelRuntime, ModelSpec  # noqa: F401

__all__ = ["incompressibleFluidBlockAMRModel", "Model", "ModelRuntime", "ModelSpec"]


@PluginSystem.register(discriminator_variable="model", discriminator="model_type")
class incompressibleFluidBlockAMRModel(BaseModel):
    """Plugin interface for solver-local incompressibleFluidBlockAMR models."""

    @classmethod
    def all_specs(cls) -> list[ModelSpec]:
        """Every registered optional-model spec, without detection (case-free)."""
        registry = PluginSystem.get_registered("incompressibleFluidBlockAMRModel")
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
        registry = PluginSystem.get_registered("incompressibleFluidBlockAMRModel")
        if not registry:
            return []

        runtimes: list[ModelRuntime] = []
        effective_dir = case_dir or Path(".")
        for plugin_cls in registry.plugin_registry:
            if not hasattr(plugin_cls, "get_model_instance"):
                continue

            spec: ModelSpec = plugin_cls.get_model_instance(plugin_cls)
            if spec.run_detect():
                runtimes.append(spec.instantiate(case_dir=effective_dir, instance_id=spec.name))

        return runtimes
