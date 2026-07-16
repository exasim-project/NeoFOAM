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

Interface (``__all__`` — grouped by role):

* **Compose & materialize** (the narrow core): :class:`Pipeline`, :class:`CaseDir`,
  :data:`Step`, :func:`empty`, :func:`from_template`. ``|`` composes,
  ``.build_at(dest)`` materializes — one meaning each; everything else is a step.
* **Steps** (each returns a :data:`Step`, injected with ``|``): :func:`patch`,
  :func:`configs`, :func:`block_mesh`, :func:`box`, :func:`snappy_hex_mesh`.
* **Read back**: :meth:`CaseDir.read_field` — a method on the case you hold, not a
  free function, so it is discoverable from the value.

Internals, off the interface but importable from their modules: the ``run_tool``
``ToolSpec`` driver behind the meshing steps (``casebuild.meshing``), the
``BoxMeshStep`` parameter model (``casebuild.meshing``), the ``pipe`` function
(a second spelling of ``|`` — ``casebuild.pipeline``), and the ``read_field`` free
function ``CaseDir.read_field`` delegates to (``casebuild.reader``).
"""

from neofoam.tooling.casebuild.meshing import block_mesh, box, snappy_hex_mesh
from neofoam.tooling.casebuild.pipeline import (
    CaseDir,
    Pipeline,
    Step,
    empty,
    from_template,
)
from neofoam.tooling.casebuild.steps import configs, patch

__all__ = [
    "CaseDir",
    "Pipeline",
    "Step",
    "empty",
    "from_template",
    "configs",
    "patch",
    "block_mesh",
    "box",
    "snappy_hex_mesh",
]
