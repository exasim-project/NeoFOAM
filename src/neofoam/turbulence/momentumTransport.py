# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The single momentum-transport plugin family.

:class:`momentumTransportModel` is the one plugin family holding every turbulence
model as a :class:`~neofoam.framework.model.ModelSpec`. Both solvers bind it, and
:func:`~neofoam.turbulence.selection.select_turbulence_model` resolves the
configured model by name and builds the handle for the requested backend
(native NeoN or pybFoam fallback — see :mod:`neofoam.turbulence.selection`).

The name follows OpenFOAM-13's ``momentumTransportModel``: it covers laminar,
RAS and LES under one honest umbrella and composes with rheology through the same
``nu``/``nut`` Context fields.

Native models register here via ``Model("name").register_with(momentumTransportModel)``.
There is **no per-model class**: a model is a :class:`ModelSpec` + config +
operations, all in one file (a native ``@build`` / native ``@operation``s and/or a
co-located ``fallback=True`` ``correct`` op).
"""

from typing import Optional

from pydantic import BaseModel

from neofoam.core.plugin_system import PluginSystem
from neofoam.framework.model import Model, ModelRuntime, ModelSpec  # noqa: F401

__all__ = [
    "momentumTransportModel",
    "Model",
    "ModelRuntime",
    "ModelSpec",
]


@PluginSystem.register(discriminator_variable="model", discriminator="model_type")
class momentumTransportModel(BaseModel):
    """Plugin interface for momentum-transport (turbulence) models."""

    @classmethod
    def all_specs(cls) -> list[ModelSpec]:
        """Return every registered momentum-transport spec, without detection."""
        registry = PluginSystem.get_registered("momentumTransportModel")
        if not registry:
            return []

        return [
            plugin_cls.get_model_instance(plugin_cls)
            for plugin_cls in registry.plugin_registry
            if hasattr(plugin_cls, "get_model_instance")
        ]

    @classmethod
    def registered_names(cls) -> list[str]:
        """Return the ``spec.name`` of every registered model."""
        return [spec.name for spec in cls.all_specs()]

    @classmethod
    def find_spec(cls, name: str) -> Optional[ModelSpec]:
        """Return the registered spec whose ``name`` matches, else ``None``."""
        for spec in cls.all_specs():
            if spec.name == name:
                return spec
        return None
