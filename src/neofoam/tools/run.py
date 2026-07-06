# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Standalone, solver-agnostic preprocessing runner.

Resolves ``system/preprocess.yaml`` against the shared tool registry and runs ONLY the
mesh DAG (+ ``_foam_time``) — no solver, fields, models, or time loop. This is the
``neofoam preprocess`` entry point. ``pyf`` is a module global so the runner is testable
without OpenFOAM.
"""

from pathlib import Path
from typing import Any, Optional

import pybFoam as pyf

from neofoam.framework.context import Context
from neofoam.framework.initialization import lazy
from neofoam.framework.initialization.execution import execute_initialization
from neofoam.framework.tools import (
    PreprocessConfig,
    ToolRuntime,
    resolve_tools,
    tool_graph_steps,
)
from neofoam.tools.registry import available_tools


def detect_tools(case_dir: Optional[Path] = None) -> list[ToolRuntime]:
    """Resolve the enable file against the shared registry (absent file → ``[]``)."""
    effective = case_dir if case_dir is not None else Path(".")
    try:
        cfg = PreprocessConfig.load(case_dir=effective)
    except FileNotFoundError:
        return []
    return resolve_tools(available_tools(), cfg)


def _case_dir_from_argv(argv: list[str]) -> str:
    """Extract the ``-case <dir>`` value from argv, defaulting to ``.``."""
    for i, token in enumerate(argv):
        if token == "-case" and i + 1 < len(argv):
            return argv[i + 1]
    return "."


def run_preprocess(argv: Optional[list[str]] = None) -> Context:
    """Build ``_foam_time`` + the resolved tool DAG and execute it; return the Context."""
    resolved_argv = argv or []

    def create_foam_time(_ctx: dict[str, Any]) -> Any:
        return pyf.Time(pyf.argList(resolved_argv))

    def read_disk_mesh(ctx: dict[str, Any]) -> Any:
        # Seeds _prev_mesh for a pipeline resuming from a mesh already on disk
        # (e.g. a single-tool snappyHexMesh slice after a prior blockMesh run).
        return pyf.fvMesh(ctx["_foam_time"])

    case_dir = Path(_case_dir_from_argv(resolved_argv))
    steps = [lazy("_foam_time", create_foam_time)]
    steps.extend(tool_graph_steps(detect_tools(case_dir), mesh_source=read_disk_mesh))
    return execute_initialization(steps)
