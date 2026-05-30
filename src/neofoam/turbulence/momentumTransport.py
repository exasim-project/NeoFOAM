# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Momentum-transport plugin interface + the one small read interface.

A momentum-transport model contributes ``divDevReff(U)`` — the divergence of the
deviatoric momentum stress. It owns the eddy viscosity ``nut`` (a Context field
it registers/updates via its operations) and **defines how its stress is
assembled**: each model registers a stress computer via
:func:`register_momentum_stress`, so a linear closure reuses
:func:`~neofoam.turbulence.stress.linear_viscous_stress` while a non-linear /
viscoelastic model registers its own — the choice stays with the model.

The name follows OpenFOAM-13's ``momentumTransportModel``: it covers laminar,
RAS and LES under one honest umbrella and composes with rheology through the same
``nu``/``nut`` Context fields.

Native models register here via ``Model("name").register_with(momentumTransportModel)``.
There is **no per-model class**: a model is a :class:`ModelSpec` + config +
operations + a registered stress computer. :class:`SpecMomentumTransport` is the
single small read interface the solver consumes; it dispatches ``divDevReff`` to
the model's registered stress.
"""

from typing import Any, Callable, Optional

from pydantic import BaseModel

from neofoam.core.plugin_system import PluginSystem
from neofoam.framework.model import Model, ModelRuntime, ModelSpec  # noqa: F401

__all__ = [
    "momentumTransportModel",
    "SpecMomentumTransport",
    "register_momentum_stress",
    "Model",
    "ModelRuntime",
    "ModelSpec",
]

#: Stress computers, registered by each model (keyed by spec name). The model
#: decides *how* ``divDevReff(U, nu, nut)`` is built; this keeps that swappable.
_STRESS: dict[str, Callable[..., Any]] = {}


def register_momentum_stress(spec: ModelSpec, stress: Callable[..., Any]) -> ModelSpec:
    """Register the stress computer a momentum-transport model dispatches to.

    Called by the model module; ``stress(U, nu, nut)`` returns the momentum
    stress matrix term. Returns ``spec`` for chaining.
    """
    _STRESS[spec.name] = stress
    return spec


class SpecMomentumTransport:
    """The one small read interface: dispatches ``divDevReff`` to the model.

    Built (in ``create_fields``) from a momentum-transport ``ModelRuntime``. The
    molecular ``nu`` and eddy ``nut`` are Context fields passed in by the
    momentum operation; this object only routes ``divDevReff`` to the model's
    registered stress computer and exposes the model's operations for the DAG.
    """

    def __init__(self, runtime: ModelRuntime) -> None:
        self._runtime = runtime
        self._stress = _STRESS[runtime.spec.name]

    def divDevReff(self, U: Any, nu: Any, nut: Any) -> Any:
        """Momentum stress term — assembled by the model's stress computer."""
        return self._stress(U, nu, nut)

    @property
    def operations(self) -> Any:
        """The model's ``@spec.operation``s, for the solver to merge into its DAG."""
        return self._runtime.operations


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
    def detect_and_create(cls) -> Any:
        """Select the active model for the case in the current directory.

        The core-model-family hook: reads ``constant/turbulenceProperties`` and
        returns the matching native spec or the OpenFOAM fallback adapter.
        """
        from .selection import select_from_case

        return select_from_case()
