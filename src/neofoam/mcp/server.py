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

Filesystem tools (``read_case``/``load_case``/``save_case``/``validate_case``/
``case_patches``) confine their path arguments under a workspace root when one is
configured (:func:`configure_root`, ``mcp serve --root``, or ``NEOFOAM_MCP_ROOT``):
paths must then be relative and any escape is rejected as a tool error. With no root
set, absolute paths are allowed — for trusted, local (stdio) use only.

This module (and :mod:`neofoam.mcp.app`) are the only ones that import the ``mcp``
extra (``fastmcp`` / ``fastapi``); without it, importing raises a plain ``ImportError`` —
install with ``pip install 'neofoam[mcp]'``.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
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
from neofoam.tooling import Workspace

#: Default ``solver`` argument for the tools (the only registered spec today).
DEFAULT_SOLVER = "incompressibleFluid"

#: The MCP server. Import this directly: ``from neofoam.mcp.server import mcp``.
mcp: FastMCP = FastMCP("neofoam")

#: Env var naming the workspace root; a fallback for the stdio entry point.
ROOT_ENV_VAR = "NEOFOAM_MCP_ROOT"

#: Process-wide workspace root confining every filesystem tool's path arguments.
#: ``None`` (unset) = trusted, absolute paths allowed (local stdio use). Set via
#: :func:`configure_root` (``mcp serve --root``) or the ``NEOFOAM_MCP_ROOT`` env var.
_root: Path | None = None


def configure_root(root: str | Path | None) -> None:
    """Set the workspace root confining every filesystem tool's ``case_dir``/``target_dir``.

    Once set, path arguments must be **relative** and are rejected (as a tool error)
    if they escape ``root`` — the trust boundary for an untrusted MCP client. ``None``
    clears it, restoring the trusted absolute-path behavior for local stdio use.
    """
    global _root
    _root = Path(root).resolve() if root is not None else None


def _workspace() -> Workspace | None:
    """The active workspace: the configured root, else ``NEOFOAM_MCP_ROOT``, else ``None``."""
    if _root is not None:
        return Workspace.at(_root)
    env = os.environ.get(ROOT_ENV_VAR)
    return Workspace.at(env) if env else None


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
    return tools.case_patches(case_dir, workspace=_workspace())


@mcp.tool
def validate_case(case_dir: str, solver: str = DEFAULT_SOLVER) -> ValidationReportDTO:
    """Static pre-flight: completeness + BC/mesh-patch + fvSolution/fvSchemes checks (no run)."""
    return tools.validate_case(resolve_solver(solver), case_dir, workspace=_workspace())


# -- case scaffolding ---------------------------------------------------------


@mcp.tool
def read_case(case_dir: str, solver: str = DEFAULT_SOLVER) -> CaseTextDTO:
    """Read every config-bound file of a case as raw text."""
    return tools.read_case(resolve_solver(solver), case_dir, workspace=_workspace())


@mcp.tool
def load_case(case_dir: str, solver: str = DEFAULT_SOLVER) -> CaseSpecDTO:
    """Load each present config from disk into an aggregate CaseSpec dump."""
    return tools.load_case(resolve_solver(solver), case_dir, workspace=_workspace())


@mcp.tool
def save_case(
    case_spec: dict[str, Any], target_dir: str, solver: str = DEFAULT_SOLVER
) -> SaveResultDTO:
    """Validate ``case_spec`` against ``solver``'s model, then write the case."""
    return tools.save_case(
        resolve_solver(solver), case_spec, target_dir, workspace=_workspace()
    )


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
