# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Standalone, solver-agnostic preprocessing runner.

Resolves ``system/preprocess.yaml`` against the shared tool registry and runs ONLY the
mesh DAG (+ ``_foam_time``) — no solver, fields, models, or time loop. This is the
``neofoam preprocess`` entry point. ``pyf`` is a module global so the runner is testable
without OpenFOAM.
"""

import os
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


def _with_absolute_case(argv: list[str], case_dir: Path) -> list[str]:
    """Rewrite the ``-case`` value to ``case_dir`` (absolute), else append it.

    We chdir into the case before running so the tools' relative ``dict_file``
    reads (``system/blockMeshDict`` …) resolve; a relative ``-case`` in argv would
    then be re-interpreted against the new cwd and break, so pin it absolute.
    """
    out = list(argv)
    for i, token in enumerate(out):
        if token == "-case" and i + 1 < len(out):
            out[i + 1] = str(case_dir)
            return out
    return out + ["-case", str(case_dir)]


def run_preprocess(argv: Optional[list[str]] = None) -> Context:
    """Build ``_foam_time`` + the resolved tool DAG and execute it; return the Context.

    Resolves ``-case`` to an absolute path and runs the DAG from inside it, so
    ``neofoam preprocess <case>`` works from any cwd (the tools read their dict
    files relative to the working directory). The original cwd is always restored.
    """
    resolved_argv = argv or []
    case_dir = Path(_case_dir_from_argv(resolved_argv)).resolve()
    foam_argv = _with_absolute_case(resolved_argv, case_dir)

    def create_foam_time(_ctx: dict[str, Any]) -> Any:
        return pyf.Time(pyf.argList(foam_argv))

    steps = [lazy("_foam_time", create_foam_time)]
    steps.extend(tool_graph_steps(detect_tools(case_dir)))
    prev_cwd = Path.cwd()
    os.chdir(case_dir)
    try:
        return execute_initialization(steps)
    finally:
        os.chdir(prev_cwd)
