# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The TimeIntegration plugin family (transient vs. steady regimes)."""

from __future__ import annotations

from typing import Any, cast

from neofoam.algorithms.time_step import MaxDeltaTConstraint
from neofoam.core.plugin_system import PluginSystem
from neofoam.algorithms.foam_time import TimeControlConfig
from neofoam.algorithms.time_integration import (
    SteadyIntegration,
    TimeIntegration,
    TransientIntegration,
    integration_from_ddt,
)


def _config(*, adjust: bool, max_delta_t: float = 2.0) -> TimeControlConfig:
    return TimeControlConfig(
        endTime=1.0,
        deltaT=0.1,
        adjustTimeStep=adjust,
        maxCo=0.5 if adjust else None,
        maxDeltaT=max_delta_t,
    )


def test_integration_from_ddt_selects_regime() -> None:
    assert isinstance(integration_from_ddt("steadyState"), SteadyIntegration)
    assert isinstance(integration_from_ddt("Euler"), TransientIntegration)
    assert isinstance(integration_from_ddt("backward"), TransientIntegration)


def test_family_is_registered_with_plugin_system() -> None:
    registry = PluginSystem.get_registered("TimeIntegration")
    assert registry is not None
    names = {c.__name__ for c in registry.plugin_registry}
    assert {"TransientIntegration", "SteadyIntegration"} <= names


def test_transient_initial_delta_t_honours_request() -> None:
    assert TransientIntegration().initial_delta_t(0.005) == 0.005


def test_steady_initial_delta_t_is_unit_step() -> None:
    assert SteadyIntegration().initial_delta_t(0.005) == 1.0


def test_transient_step_name_is_general_float() -> None:
    assert TransientIntegration().step_name(0.005, 1, 6) == "0.005"


def test_steady_step_name_is_integer_iteration_index() -> None:
    assert SteadyIntegration().step_name(3.0, 3, 6) == "3"


def test_transient_constraints_gate_on_adjust_time_step() -> None:
    assert TransientIntegration().constraints(_config(adjust=False)) == []

    cons = TransientIntegration().constraints(_config(adjust=True, max_delta_t=2.0))
    assert len(cons) == 1
    assert isinstance(cons[0], MaxDeltaTConstraint)
    assert cons[0].maxDeltaT == 2.0


def test_steady_constraints_are_empty() -> None:
    assert SteadyIntegration().constraints(_config(adjust=True)) == []


def test_selectable_through_discriminated_union() -> None:
    # registration alone makes a regime selectable by its discriminator keyword
    create = cast(Any, TimeIntegration).create
    wrapper = create(integration={"time_integration_type": "steady"})
    assert isinstance(wrapper.integration, SteadyIntegration)
    wrapper = create(integration={"time_integration_type": "transient"})
    assert isinstance(wrapper.integration, TransientIntegration)
