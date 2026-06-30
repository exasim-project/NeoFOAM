# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

import asyncio
import json

import pytest

pytest.importorskip("fastmcp")
pytest.importorskip("pybFoam")

from neofoam.mcp import tools  # noqa: E402
from neofoam.mcp.registry import resolve_solver  # noqa: E402
from neofoam.mcp.server import (  # noqa: E402
    DEFAULT_SOLVER,
    mcp,
    model_catalog,
    registered_tool_names,
)
from neofoam.mcp.tools import ALL_TOOL_NAMES  # noqa: E402


def _resource_json(uri: str):
    async def _read():
        return await mcp.read_resource(uri)

    result = asyncio.run(_read())
    return json.loads(result.contents[0].content)


def test_registered_tools_are_the_full_set_and_no_run() -> None:
    names = registered_tool_names()
    assert names == set(ALL_TOOL_NAMES)
    # No tool that runs/solves a case is exposed (repeated in-process pybFoam
    # runs segfault); the surface must contain none of these.
    forbidden = {"run", "solve", "run_case", "run_solver", "solve_case"}
    assert not (names & forbidden)


def test_tool_takes_solver_argument_with_default() -> None:
    # @mcp.tool leaves the function callable; the solver arg defaults to the
    # registered default and accepts an explicit registered name.
    assert DEFAULT_SOLVER == "incompressibleFluid"
    default = {e.name for e in model_catalog()}
    explicit = {e.name for e in model_catalog(solver="incompressibleFluid")}
    assert default == explicit
    assert {"courant", "maxDeltaT"} <= default


def test_tool_unknown_solver_raises() -> None:
    # An unknown solver name fails fast (surfaced to the client as a tool error).
    with pytest.raises(ValueError):
        model_catalog(solver="nope")


def test_catalog_resource_mirrors_model_catalog() -> None:
    expected = [
        d.model_dump()
        for d in tools.model_catalog(resolve_solver("incompressibleFluid"))
    ]
    assert _resource_json("neofoam://incompressibleFluid/catalog") == expected


def test_solvers_resource_lists_known_solvers() -> None:
    assert "incompressibleFluid" in _resource_json("neofoam://solvers")


def test_config_schema_resource_returns_schema() -> None:
    payload = _resource_json(
        "neofoam://incompressibleFluid/config/ControlDictConfig/schema"
    )
    assert payload["name"] == "ControlDictConfig"
    assert "properties" in payload["json_schema"]
