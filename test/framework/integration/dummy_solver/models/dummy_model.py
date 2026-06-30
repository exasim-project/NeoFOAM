# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Base class for DummySolver models.

Mimics SimpleSolverModel structure for testing.
"""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from neofoam.core.plugin_system import PluginSystem
from neofoam.framework.model import ModelRuntime, ModelSpec


def Model(name: str) -> ModelSpec:
    """Factory for dummy models using the ModelSpec API."""
    return ModelSpec(name)


@PluginSystem.register(discriminator_variable="model", discriminator="model_type")
class DummyModelInterface(BaseModel):
    """
    Base class for DummySolver optional models.

    Implements the *optional model family* contract consumed by
    ``SolverSpec.models(...)``: ``all_specs()`` lists every
    registered member case-free (for the config schema) and
    ``detect_models()`` returns the active members for a concrete case.
    """

    @classmethod
    def _registered_specs(cls) -> list[ModelSpec]:
        registry = PluginSystem.get_registered("DummyModelInterface")
        if not registry:
            return []
        return [
            plugin_cls.get_model_instance(plugin_cls)
            for plugin_cls in registry.plugin_registry
            if hasattr(plugin_cls, "get_model_instance")
        ]

    @classmethod
    def all_specs(cls) -> list[ModelSpec]:
        """Every registered member spec, case-free (no detection).

        Lists the full set of optional models so their config classes can be
        enumerated by :func:`neofoam.configurations` without a case.
        """
        return cls._registered_specs()

    @classmethod
    def detect_specs(cls) -> list[ModelSpec]:
        """Registered ModelSpec objects that pass their ``detect()`` check.

        Callers call ``spec.instantiate(case_dir, instance_id)`` per config entry.
        """
        return [spec for spec in cls._registered_specs() if spec.run_detect()]

    @classmethod
    def detect_models(cls, case_dir: Optional[Path] = None) -> list[ModelRuntime]:
        """Instantiate the detected members into runtimes for a concrete case."""
        effective_dir = case_dir or Path(__file__).parent.parent / "configs"
        return [
            spec.instantiate(case_dir=effective_dir, instance_id=spec.name)
            for spec in cls.detect_specs()
        ]
