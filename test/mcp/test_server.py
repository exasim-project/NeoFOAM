# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

import asyncio
import json
from pathlib import Path

import pytest

pytest.importorskip("fastmcp")

from neofoam.mcp import tools  # noqa: E402
from neofoam.mcp.registry import resolve_solver  # noqa: E402
from neofoam.mcp.server import (  # noqa: E402
    DEFAULT_SOLVER,
    ROOT_ENV_VAR,
    configure_root,
    mcp,
    model_catalog,
    read_case,
    registered_tool_names,
)
from neofoam.mcp.tools import ALL_TOOL_NAMES  # noqa: E402
from neofoam.tooling import CaseAccessError  # noqa: E402


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


def test_registered_tools_carry_grouping_tags_and_meta() -> None:
    # Curated tags/meta sharpen tool selection + let a client filter by group.
    registered = asyncio.run(mcp.list_tools())
    by_name = {t.name: t for t in registered}
    assert "introspection" in by_name["model_catalog"].tags
    assert "geometry" in by_name["import_geometry"].tags
    assert "workspace" in by_name["workspace_info"].tags
    # every tool carries the shared provenance meta
    for tool in registered:
        assert (tool.meta or {}).get("package") == "neofoam"


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
    assert payload["name"] == "control_dict_config"
    assert "properties" in payload["json_schema"]


@pytest.fixture
def confined_root(tmp_path: Path):
    """Configure (and tear down) a server-wide workspace root at ``tmp_path``.

    The root confines every filesystem tool; reset to ``None`` afterwards so the
    module-level state does not leak into the unconfined tests above.
    """
    configure_root(tmp_path)
    try:
        yield tmp_path
    finally:
        configure_root(None)


def test_configured_root_rejects_an_escaping_case_dir(confined_root: Path) -> None:
    # With a root set, the server-level tool (not the plain tools.* fn) confines the
    # path: an absolute/escaping case_dir is a tool error, so no read happens.
    with pytest.raises(CaseAccessError):
        read_case("../escape")


def test_configured_root_resolves_a_relative_case_dir(confined_root: Path) -> None:
    # A relative case_dir under the root is resolved (not rejected as an escape); a
    # missing case then fails on absence — the "does not exist" branch, proving
    # confinement passed the path through rather than bouncing it as an escape.
    with pytest.raises(CaseAccessError, match="does not exist"):
        read_case("no_such_case")


def test_root_env_var_confines_when_no_explicit_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # NEOFOAM_MCP_ROOT is the stdio-entry fallback when configure_root was not called.
    monkeypatch.setenv(ROOT_ENV_VAR, str(tmp_path))
    with pytest.raises(CaseAccessError):
        read_case("/etc/passwd")
