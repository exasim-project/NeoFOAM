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

Native models register here via
``register_momentum_transport(Model("name"), family="RAS")``. There is **no
per-model class**: a model is a :class:`ModelSpec` + config + operations, all in
one file (a native ``@build`` / native ``@operation``s and/or a co-located
``fallback=True`` ``correct`` op).

Registration also records the *turbulence family* the closure belongs to
(``laminar`` / ``RAS`` / ``LES``), so the selector can refuse to build a RAS
closure for an ``LES { LESModel … }`` entry (and vice versa) — a name alone does
not identify a model, OpenFOAM keeps a separate run-time selection table per
family.
"""

from typing import Literal, Optional

from pydantic import BaseModel

from neofoam.core.plugin_system import PluginSystem
from neofoam.framework.model import Model, ModelRuntime, ModelSpec  # noqa: F401

__all__ = [
    "momentumTransportModel",
    "register_momentum_transport",
    "TurbulenceFamily",
    "Model",
    "ModelRuntime",
    "ModelSpec",
]

#: The turbulence families OpenFOAM selects a momentum-transport closure from —
#: the same closed set as ``turbulenceProperties/simulationType``.
TurbulenceFamily = Literal["laminar", "RAS", "LES"]

#: Family of each registered model, by spec name. Kept beside the registry rather
#: than on the generic ``ModelSpec``: the family is turbulence-specific metadata.
_MODEL_FAMILIES: dict[str, TurbulenceFamily] = {}


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

    @classmethod
    def family_of(cls, name: str) -> Optional[TurbulenceFamily]:
        """Return the family (``laminar``/``RAS``/``LES``) *name* was registered as.

        ``None`` for a name that is not registered.
        """
        return _MODEL_FAMILIES.get(name)


def register_momentum_transport(spec: ModelSpec, *, family: TurbulenceFamily) -> ModelSpec:
    """Register *spec* as a momentum-transport model of *family*.

    Use in place of ``spec.register_with(momentumTransportModel)`` so the model
    declares which OpenFOAM selection table it is a closure for; the selector
    rejects a spec whose family differs from the case's ``simulationType``.

    Example:
        ``kEpsilon = register_momentum_transport(Model("kEpsilon"), family="RAS")``
    """
    _MODEL_FAMILIES[spec.name] = family
    return spec.register_with(momentumTransportModel)
