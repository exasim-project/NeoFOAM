# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The TimeIntegration plugin family (transient vs. steady regimes)."""

from __future__ import annotations

from typing import Any, cast

from neofoam.core.plugin_system import PluginSystem
from neofoam.algorithms.solution_loop.time_integration import (
    SteadyIntegration,
    TimeIntegration,
    TransientIntegration,
    integration_from_ddt,
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


def test_selectable_through_discriminated_union() -> None:
    # registration alone makes a regime selectable by its discriminator keyword
    create = cast(Any, TimeIntegration).create
    wrapper = create(integration={"time_integration_type": "steady"})
    assert isinstance(wrapper.integration, SteadyIntegration)
    wrapper = create(integration={"time_integration_type": "transient"})
    assert isinstance(wrapper.integration, TransientIntegration)
