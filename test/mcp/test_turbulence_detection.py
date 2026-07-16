# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""MCP turbulence-model detection for both incompressible solvers.

After the family merge every turbulence model is a member of the single
``momentumTransportModel`` family, which **both** ``incompressibleFluid`` and
``incompressibleFluidNeoN`` bind. So every registered model surfaces in the MCP
``model_catalog`` for both solvers, and registering a *new* model makes it
discoverable with no MCP-code change — the behavior these tests pin.
"""

from typing import Iterator

import pytest

from neofoam.core.plugin_system import PluginSystem
from neofoam.mcp import tools
from neofoam.mcp.registry import resolve_solver
from neofoam.turbulence import momentumTransportModel
from neofoam.turbulence.momentumTransport import Model

_BUNDLED = {"laminar", "kEpsilon", "kOmegaSST", "SpalartAllmaras", "realizableKE"}


def _catalog_names(solver_name: str) -> set[str]:
    return {m.name for m in tools.model_catalog(resolve_solver(solver_name))}


@pytest.mark.parametrize(
    "solver_name", ["incompressibleFluid", "incompressibleFluidNeoN"]
)
def test_bundled_turbulence_models_are_listed(solver_name: str) -> None:
    assert _BUNDLED <= _catalog_names(solver_name)


@pytest.fixture
def throwaway_model() -> Iterator[str]:
    """Register a probe model, then remove it from the family registry."""
    from neofoam.turbulence.config import TurbulencePropertiesConfig

    registry = PluginSystem.get_registered("momentumTransportModel")
    assert registry is not None
    before = list(registry.plugin_registry)

    probe = Model("mcpProbe").register_with(momentumTransportModel)
    probe.config(TurbulencePropertiesConfig)
    try:
        yield "mcpProbe"
    finally:
        # Drop only what our registration added (register_with appends one wrapper).
        registry.plugin_registry[:] = [
            p for p in registry.plugin_registry if p in before
        ]


@pytest.mark.parametrize(
    "solver_name", ["incompressibleFluid", "incompressibleFluidNeoN"]
)
def test_newly_registered_model_surfaces_out_of_the_box(
    solver_name: str, throwaway_model: str
) -> None:
    assert throwaway_model in _catalog_names(solver_name)


def test_probe_is_gone_after_teardown() -> None:
    # Runs without the fixture: the probe must not have leaked into the registry.
    assert "mcpProbe" not in momentumTransportModel.registered_names()
