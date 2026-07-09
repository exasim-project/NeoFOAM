# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Alpha-advection plugin interface (family of interchangeable schemes).

The phase-fraction transport scheme is a *core model family*: exactly one member
is active per case. Members register here via
``Model("name").register_with(advectionModel)`` and are discovered case-free by
name; the active one is chosen from the ``advectionScheme`` key in
``system/fvSolution`` (default ``MULES``) — see :mod:`selection`.

Mirrors :mod:`neofoam.viscosity.viscosityModel`. Each member is a
:class:`ModelSpec` + operations that owns the shared VoF fields (via
``shared_field_build_steps``) and exposes an ``alpha_advection`` operation, so
the solver entrypoint stays agnostic to which scheme is active.
"""

from typing import Any, Optional

from pydantic import BaseModel

from neofoam.core.plugin_system import PluginSystem
from neofoam.framework.model import Model, ModelRuntime, ModelSpec  # noqa: F401

__all__ = ["advectionModel", "Model", "ModelRuntime", "ModelSpec"]


@PluginSystem.register(discriminator_variable="model", discriminator="model_type")
class advectionModel(BaseModel):
    """Plugin interface for alpha (phase-fraction) advection schemes."""

    @classmethod
    def all_specs(cls) -> list[ModelSpec]:
        """Return every registered advection spec, without detection."""
        registry = PluginSystem.get_registered("advectionModel")
        if not registry:
            return []

        return [
            plugin_cls.get_model_instance(plugin_cls)
            for plugin_cls in registry.plugin_registry
            if hasattr(plugin_cls, "get_model_instance")
        ]

    @classmethod
    def registered_names(cls) -> list[str]:
        """Return the ``spec.name`` of every registered advection scheme."""
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
        """Select the active advection scheme for the case in the cwd.

        The core-model-family hook: reads the ``advectionScheme`` key from
        ``system/fvSolution`` and returns the matching registered spec (MULES
        fallback on unknown/absent).
        """
        from .selection import select_from_case

        return select_from_case()
