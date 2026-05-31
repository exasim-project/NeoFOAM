# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Momentum-transport plugin interface + the one small read interface.

A momentum-transport model contributes ``divDevReff(U)`` — the divergence of the
deviatoric momentum stress — and **defines which stress it uses**: each concrete
model returns its own viscous-stress object from :meth:`viscous_stress`. The
family interface :class:`momentumTransportModel` and the wrapper
:class:`SpecMomentumTransport` are only *dispatchers* — they select / forward to
the concrete model (laminar, kEpsilon, …, or the OpenFOAM fallback), they do not
decide the stress themselves.

The name follows OpenFOAM-13's ``momentumTransportModel``: it covers laminar,
RAS and LES under one honest umbrella and composes with rheology through the same
``nu``/``nut`` Context fields.

Native models register here via ``Model("name").register_with(momentumTransportModel)``.
There is **no per-model class**: a model is a :class:`ModelSpec` + config +
operations. :class:`SpecMomentumTransport` is the single small read interface the
solver consumes for native models; the OpenFOAM fallback
(:class:`~neofoam.turbulence.fallback.OpenFOAMTurbulenceModel`) is a peer model
with the same interface, so the solver treats both uniformly.
"""

from typing import Any, Optional

from pydantic import BaseModel

from neofoam.core.plugin_system import PluginSystem
from neofoam.framework.model import Model, ModelRuntime, ModelSpec  # noqa: F401

from .stress import LinearViscousStress

__all__ = [
    "momentumTransportModel",
    "SpecMomentumTransport",
    "Model",
    "ModelRuntime",
    "ModelSpec",
]


class SpecMomentumTransport:
    """A native momentum-transport model placed at ``models.turbulence``.

    Built (in ``create_fields``) from a momentum-transport ``ModelRuntime``. It
    owns the eddy viscosity ``nut`` through the runtime's operations and
    **defines its momentum stress** via :meth:`viscous_stress`: native linear
    closures (laminar, kEpsilon, kOmegaSST, Smagorinsky, …) use the linear
    eddy-viscosity assembly (``nuEff = nu + nut``). A non-linear / viscoelastic
    native model would override this to return its own stress.
    """

    #: Descriptive tag for the stress family this model uses.
    stress_kind = "linear"

    def __init__(self, runtime: ModelRuntime) -> None:
        self._runtime = runtime

    @property
    def operations(self) -> Any:
        """The model's ``@spec.operation``s, for the solver to merge into its DAG."""
        return self._runtime.operations

    def viscous_stress(self) -> LinearViscousStress:
        """The stress this model uses — the linear eddy-viscosity assembly."""
        return LinearViscousStress()


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
