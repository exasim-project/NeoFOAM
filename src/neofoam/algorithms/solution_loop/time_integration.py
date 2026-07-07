# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""How a step advances — a plugin family of time-integration regimes.

Whether each step is a slice of *physical time* (transient) or a unit
*iteration* (steady relaxation) is neither the stepper's nor the loop's decision
to hard-code. :class:`TimeIntegration` is a
:class:`~neofoam.core.plugin_system.PluginSystem` family: each regime registers
itself (discriminated by ``time_integration_type``) and owns the three
regime-specific decisions the stepper used to branch on with an ``enum`` —

* ``initial_delta_t`` — the starting step (a transient run honours ``deltaT``; a
  steady run is a unit pseudo-step);
* ``step_name`` — how the current step renders (a float time vs. an integer
  iteration index).

A new regime (pseudo-transient, local/dual time stepping, …) is a new
``@TimeIntegration.register`` class — never an edit to a branch (OCP). The data
that *selects* the regime is the case's ``ddtSchemes`` default
(``steadyState`` ⇒ steady), mapped by :func:`integration_from_ddt`.
"""

from __future__ import annotations

from typing import Literal

from neofoam.core.plugin_system import PluginSystem
from neofoam.io import BaseConfig


@PluginSystem.register(
    discriminator_variable="integration", discriminator="time_integration_type"
)
class TimeIntegration(BaseConfig):
    """Plugin interface: the regime-specific decisions of one advancement step.

    Concrete regimes register with :meth:`TimeIntegration.register`; the
    ``time_integration_type`` literal discriminates them. The stepper holds one
    and delegates, so it carries no ``time``-vs-``iteration`` branch.
    """

    def initial_delta_t(self, requested: float) -> float:
        raise NotImplementedError

    def step_name(self, value: float, index: int, precision: int) -> str:
        raise NotImplementedError


@TimeIntegration.register
class TransientIntegration(TimeIntegration):
    """``ddtScheme`` Euler/backward/CrankNicolson — advance physical time, ``deltaT``-sized."""

    time_integration_type: Literal["transient"] = "transient"

    def initial_delta_t(self, requested: float) -> float:
        return requested

    def step_name(self, value: float, index: int, precision: int) -> str:
        # general float format (trims trailing zeros, like %g)
        return f"{value:.{precision}g}"


@TimeIntegration.register
class SteadyIntegration(TimeIntegration):
    """``ddtScheme steadyState`` — unit step; "time" is the iteration index, fixed step."""

    time_integration_type: Literal["steady"] = "steady"

    def initial_delta_t(self, requested: float) -> float:
        return 1.0

    def step_name(self, value: float, index: int, precision: int) -> str:
        return str(int(round(value)))


def integration_from_ddt(ddt_default: str) -> TimeIntegration:
    """``ddtSchemes`` default ``steadyState`` ⇒ steady (iteration), else transient."""
    if ddt_default.strip() == "steadyState":
        return SteadyIntegration()
    return TransientIntegration()
