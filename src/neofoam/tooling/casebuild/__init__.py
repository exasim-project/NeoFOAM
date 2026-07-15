# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Pipe-composed OpenFOAM case construction.

Describe a case as a start state (:func:`empty` / :func:`from_template`) composed with
ordered steps — a meshing strategy (:func:`block_mesh`, :func:`box`,
:func:`snappy_hex_mesh`), :func:`patch`, :func:`configs` — via ``|``, then materialize
it with ``.build_at(dest)``::

    case = (from_template(src) | block_mesh() | patch("system/controlDict", endTime=0.1)).build_at(tmp)

A :class:`Pipeline` is a value — nothing touches disk until ``.build_at()`` — so it can be
materialized repeatedly, and ``base | step`` forks a materialized :class:`CaseDir`.
"""

from neofoam.tooling.casebuild.meshing import (
    BoxMeshStep,
    block_mesh,
    box,
    run_tool,
    snappy_hex_mesh,
)
from neofoam.tooling.casebuild.pipeline import (
    CaseDir,
    Pipeline,
    Step,
    empty,
    from_template,
    pipe,
)
from neofoam.tooling.casebuild.reader import read_field
from neofoam.tooling.casebuild.steps import configs, patch

__all__ = [
    "CaseDir",
    "Pipeline",
    "Step",
    "empty",
    "from_template",
    "pipe",
    "configs",
    "patch",
    "block_mesh",
    "box",
    "read_field",
    "run_tool",
    "snappy_hex_mesh",
    "BoxMeshStep",
]
