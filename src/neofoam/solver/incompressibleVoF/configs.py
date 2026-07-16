# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Solver-core configs for incompressibleVoF (interFoam-style two-phase VoF).

These pydantic ``BaseConfig`` classes declare the case-authoring surface the
solver's C++ backend reads at run time but that no model family already owns:

- :class:`ControlDictConfig` — ``system/controlDict`` with the interFoam
  adaptive-stepping keys (``adjustTimeStep`` / ``maxCo`` / ``maxAlphaCo`` /
  ``maxDeltaT``) that ``set_time_step`` re-reads every step.
- :class:`TransportPropertiesConfig` — the **two-phase** ``constant/
  transportProperties`` (``phases`` + per-phase ``transportModel`` / ``nu`` /
  ``rho`` + ``sigma``) that the C++ ``immiscibleIncompressibleTwoPhaseMixture``
  constructor reads. Single-phase transport configs
  (:class:`neofoam.viscosity.config.TransportPropertiesConfig`,
  ``BoussinesqConfig``) do not model ``phases``/``sigma``, so VoF needs its own.
- :class:`GravityConfig` — ``constant/g`` (buoyant pressure needs gravity).

:class:`~neofoam.turbulence.config.TurbulencePropertiesConfig` is re-exported so
the solver can declare it without importing the turbulence package by path; the
two-phase ``TwoPhaseTransportModel`` reads ``constant/turbulenceProperties``.

None of these change the solve: they are declared on the solver spec purely so
``configurations(incompressibleVoF)`` / the case wizard can fill and persist the
files the C++ backend then reads.
"""

from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
)

from neofoam.algorithms.solution_loop.config import TimeControlConfig
from neofoam.io import OF, BaseConfig, IOStrategy
from neofoam.turbulence.config import TurbulencePropertiesConfig

__all__ = [
    "ControlDictConfig",
    "TransportPropertiesConfig",
    "GravityConfig",
    "TurbulencePropertiesConfig",
]


@IOStrategy(OF("system/controlDict"))
class ControlDictConfig(TimeControlConfig):
    """``system/controlDict`` — the file-bound time/write config for VoF.

    Inherits the time-stepping + write schema (``startTime`` / ``endTime`` /
    ``deltaT`` / ``writeControl`` / ``writeInterval``) from
    :class:`~neofoam.algorithms.solution_loop.config.TimeControlConfig`.

    Unlike ``incompressibleFluid`` (which co-owns the adaptive keys through
    opt-in ``courant`` / ``max_delta_t`` models), VoF's ``set_time_step`` re-reads
    ``adjustTimeStep`` / ``maxCo`` / ``maxAlphaCo`` / ``maxDeltaT`` from
    ``controlDict`` every step (the dual flow/interface Courant limiter is core to
    interFoam), so they live here with interFoam-typical defaults.
    """

    application: str = "interFoam"
    adjustTimeStep: bool = False
    maxCo: float = 1.0
    maxAlphaCo: float = 1.0
    maxDeltaT: float = 1.0


class PhaseTransport(BaseModel):
    """One phase's transport sub-dict in ``constant/transportProperties``.

    ``transportModel Newtonian; nu <kinematic viscosity>; rho <density>;`` — the
    interFoam per-phase block. ``nu`` is a kinematic viscosity (m^2/s), ``rho`` a
    density (kg/m^3).
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    transportModel: str = "Newtonian"
    nu: float
    rho: float


@IOStrategy(OF("constant/transportProperties"))
class TransportPropertiesConfig(BaseConfig):
    """``constant/transportProperties`` — the interFoam two-phase transport.

    Models the file the C++ ``immiscibleIncompressibleTwoPhaseMixture`` reads:
    the ordered ``phases`` list, one transport sub-dict per phase, and the
    surface-tension coefficient ``sigma``. Defaults describe water flooding an
    air-filled domain (the hx-cad VoF study).
    """

    phases: list[str] = Field(default_factory=lambda: ["water", "air"])
    water: PhaseTransport = Field(
        default_factory=lambda: PhaseTransport(nu=1e-6, rho=1000.0)
    )
    air: PhaseTransport = Field(
        default_factory=lambda: PhaseTransport(nu=1.48e-5, rho=1.0)
    )
    sigma: float = 0.07

    @field_serializer("phases", when_used="always")
    def _serialize_phases(self, value: list[str]) -> str:
        return "(" + " ".join(value) + ")"

    @field_validator("phases", mode="before")
    @classmethod
    def _parse_phases(cls, value: Any) -> list[str]:
        if isinstance(value, str):
            return value.strip().strip("()").split()
        return list(value)


class _GravityHeader(BaseModel):
    """The ``FoamFile`` header that identifies ``constant/g`` to OpenFOAM.

    ``class`` is pinned to ``uniformDimensionedVectorField`` so OpenFOAM reads the
    file as a gravity field (not the generic ``dictionary`` the writer injects for
    headerless configs).
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    version: str = "2.0"
    format: str = "ascii"
    field_class: str = Field(default="uniformDimensionedVectorField", alias="class")
    object: str = "g"


def _fmt_component(x: float) -> str:
    """Render a vector component with no gratuitous ``.0`` (``0.0`` -> ``0``)."""
    return str(int(x)) if x == int(x) else repr(x)


@IOStrategy(OF("constant/g"))
class GravityConfig(BaseConfig):
    """``constant/g`` — the uniform gravitational acceleration.

    The VoF ``p_rgh`` formulation reads ``g`` at init to form the geopotential
    ``gh`` / ``ghf``; without this file the solver aborts with *cannot find file
    "constant/g"*. Defaults are Earth gravity acting in -y: ``dimensions
    [0 1 -2 0 0 0 0]``, ``value (0 -9.81 0)``.

    A field-for-field copy of ``incompressibleFluid``'s ``GravityConfig``; the two
    are unified in a neutral module as a follow-up (the VoF review branch keeps its
    solver package self-contained).
    """

    FoamFile: _GravityHeader = Field(default_factory=_GravityHeader)
    dimensions: list[int] = Field(default_factory=lambda: [0, 1, -2, 0, 0, 0, 0])
    value: list[float] = Field(default_factory=lambda: [0.0, -9.81, 0.0])

    @field_validator("dimensions", mode="before")
    @classmethod
    def _parse_dimensions(cls, value: Any) -> list[int]:
        if isinstance(value, str):
            return [int(p) for p in value.strip().strip("[]").split()]
        return [int(v) for v in value]

    @field_validator("value", mode="before")
    @classmethod
    def _parse_value(cls, value: Any) -> list[float]:
        if isinstance(value, str):
            return [float(p) for p in value.strip().strip("()").split()]
        return [float(v) for v in value]

    @field_serializer("dimensions", when_used="always")
    def _serialize_dimensions(self, value: list[int]) -> str:
        return "[" + " ".join(str(int(v)) for v in value) + "]"

    @field_serializer("value", when_used="always")
    def _serialize_value(self, value: list[float]) -> str:
        return "(" + " ".join(_fmt_component(v) for v in value) + ")"
