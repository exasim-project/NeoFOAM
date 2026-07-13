# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The MCP server — a module-level ``mcp`` (FastMCP v3) with every tool/resource.

Built with the FastMCP decorator pattern at module level: ``mcp`` is the server,
each ``@mcp.tool`` / ``@mcp.resource`` is a thin wrapper over the plain logic in
:mod:`neofoam.mcp.tools`. Import it directly — ``from neofoam.mcp.server import mcp``.
Every tool takes a ``solver`` **argument** (a registered spec name, default
``incompressibleFluid``) and resolves it per call, so the server is solver-agnostic;
an unknown name raises ``ValueError`` (surfaced as a tool error). The
``neofoam://{solver_name}/…`` resources resolve by name the same way.

This module (and :mod:`neofoam.mcp.app`) are the only ones that import the ``mcp``
extra (``fastmcp`` / ``fastapi``); without it, importing raises a plain ``ImportError`` —
install with ``pip install 'neofoam[mcp]'``.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastmcp import FastMCP

from neofoam.mcp import tools
from neofoam.mcp.dto import (
    CaseSpecDTO,
    CaseTextDTO,
    ConfigInfoDTO,
    ConfigSchemaDTO,
    ModelEntryDTO,
    PatchDTO,
    SaveResultDTO,
    ToolInfoDTO,
    ValidationReportDTO,
)
from neofoam.mcp.registry import resolve_solver

#: Default ``solver`` argument for the tools (the only registered spec today).
DEFAULT_SOLVER = "incompressibleFluid"

#: The MCP server. Import this directly: ``from neofoam.mcp.server import mcp``.
mcp: FastMCP = FastMCP("neofoam")


# -- introspection (case-free) ------------------------------------------------


@mcp.tool
def list_solvers() -> list[str]:
    """Known solver spec names."""
    return tools.list_solvers()


@mcp.tool
def model_catalog(solver: str = DEFAULT_SOLVER) -> list[ModelEntryDTO]:
    """Every model of ``solver`` with its required flag + owned configs."""
    return tools.model_catalog(resolve_solver(solver))


@mcp.tool
def tool_catalog(solver: str = DEFAULT_SOLVER) -> list[ToolInfoDTO]:
    """Preprocessing tools (blockMesh/snappyHexMesh/checkMesh) + the config each reads."""
    return tools.tool_catalog(resolve_solver(solver))


@mcp.tool
def list_configs(solver: str = DEFAULT_SOLVER) -> list[ConfigInfoDTO]:
    """Every config class ``solver`` may consume (name/cls_name/file)."""
    return tools.list_configs(resolve_solver(solver))


@mcp.tool
def config_schema(name: str, solver: str = DEFAULT_SOLVER) -> ConfigSchemaDTO:
    """JSON Schema + rjsf ui-schema + defaults for one config class of ``solver``."""
    return tools.config_schema(resolve_solver(solver), name)


# -- case geometry ------------------------------------------------------------


@mcp.tool
def case_patches(case_dir: str) -> list[PatchDTO]:
    """Boundary patches (name + role) of a staged case, for authoring boundary conditions."""
    return tools.case_patches(case_dir)


@mcp.tool
def validate_case(case_dir: str, solver: str = DEFAULT_SOLVER) -> ValidationReportDTO:
    """Static pre-flight: completeness + BC/mesh-patch + fvSolution/fvSchemes checks (no run)."""
    return tools.validate_case(resolve_solver(solver), case_dir)


# -- case scaffolding ---------------------------------------------------------


@mcp.tool
def read_case(case_dir: str, solver: str = DEFAULT_SOLVER) -> CaseTextDTO:
    """Read every config-bound file of a case as raw text."""
    return tools.read_case(resolve_solver(solver), case_dir)


@mcp.tool
def load_case(case_dir: str, solver: str = DEFAULT_SOLVER) -> CaseSpecDTO:
    """Load each present config from disk into an aggregate CaseSpec dump."""
    return tools.load_case(resolve_solver(solver), case_dir)


@mcp.tool
def save_case(
    case_spec: dict[str, Any], target_dir: str, solver: str = DEFAULT_SOLVER
) -> SaveResultDTO:
    """Validate ``case_spec`` against ``solver``'s model, then write the case."""
    return tools.save_case(resolve_solver(solver), case_spec, target_dir)


# -- read-only resources mirroring the introspection tools --------------------
# Each returns a JSON string (one ``text/plain`` content); a bare ``list[dict]``
# return is otherwise read by FastMCP as *multiple* contents.


@mcp.resource("neofoam://solvers")
def solvers_resource() -> str:
    return json.dumps(tools.list_solvers())


@mcp.resource("neofoam://{solver_name}/catalog")
def catalog_resource(solver_name: str) -> str:
    entries = tools.model_catalog(resolve_solver(solver_name))
    return json.dumps([d.model_dump() for d in entries])


@mcp.resource("neofoam://{solver_name}/config/{name}/schema")
def config_schema_resource(solver_name: str, name: str) -> str:
    return tools.config_schema(resolve_solver(solver_name), name).model_dump_json()


def registered_tool_names(server: FastMCP = mcp) -> set[str]:
    """The registered tool names via the public FastMCP API."""
    return {tool.name for tool in asyncio.run(server.list_tools())}


if __name__ == "__main__":
    # ``python -m neofoam.mcp.server`` — serve over stdio (default transport), for an
    # MCP client (Claude Code, etc.) to launch as a subprocess.
    mcp.run()
