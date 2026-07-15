# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Injectable meshing strategies as pipeline steps: ``block_mesh``, ``box``, ``snappy_hex_mesh``.

There is deliberately no default meshing verb — a case's mesh is chosen by injecting
one of these steps. Each reuses the shared ``neofoam.tools`` mesh tools
(``blockMeshTool`` / ``snappyHexMeshTool`` + their pydantic step configs) rather than
re-wrapping ``pybFoam.meshing``: :func:`run_tool` drives a tool's ``InitStep`` objects
directly against a :class:`CaseDir`, seeding the ``ctx`` the tool expects
(``_foam_time``, and ``_prev_mesh`` for mesh-consuming tools). A tool's ``dict_file``
option is absolutized against the case so the Foam ``Time`` can use the two-arg
(root, case) form — no ``chdir`` and no ``argList`` lifetime hazard.

:func:`run_tool` is the single in-process tool engine: these steps and the sweep
runner (:mod:`neofoam.tooling.workflow.sweep_runner`) both drive mesh tools through it.

``block_mesh`` reads a committed ``blockMeshDict``; ``box`` synthesizes one from
``n``/``dims`` (the OpenFOAM ``blocks``/``boundary`` grammar can't be built through the
typed ``dictionary`` API, so it is emitted as text); ``snappy_hex_mesh`` refines the mesh
a prior ``block_mesh`` left on disk (reconstructed via ``pyf.fvMesh``).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

import pybFoam as pyf
from pydantic import BaseModel

from neofoam.framework.tools.spec import ToolSpec
from neofoam.tooling.casebuild.pipeline import CaseDir, Step
from neofoam.tools.block_mesh import blockMeshTool
from neofoam.tools.snappy_hex_mesh import snappyHexMeshTool


def seed_tool_ctx(case: CaseDir, *, needs_prev_mesh: bool) -> dict[str, Any]:
    """Build the init ``ctx`` a tool's ``@build`` reads for an in-process run.

    The single place the Foam objects are constructed for a one-shot tool run: a
    two-arg (root, case) ``pyf.Time`` seeded as ``_foam_time`` (so no ``chdir`` and no
    ``argList`` lifetime hazard), plus — when *needs_prev_mesh* — ``_prev_mesh``
    reconstructed via ``pyf.fvMesh`` from the on-disk mesh a prior ``block_mesh`` left,
    the input a refiner (snappy) or validator (checkMesh) reads.

    This is deliberately NOT shared with :func:`neofoam.framework.tools.graph.tool_graph_steps`:
    that engine is a DAG *chainer* that receives ``_foam_time`` from the solver Context
    and threads ``_prev_mesh`` between ``InitStep`` outputs — it never constructs a
    ``Time``/``fvMesh`` itself, so there is nothing to fold together.
    """
    time = pyf.Time(str(case.path.parent), case.path.name)
    ctx: dict[str, Any] = {"_foam_time": time}
    if needs_prev_mesh:
        ctx["_prev_mesh"] = pyf.fvMesh(time)
    return ctx


def run_tool(
    case: CaseDir,
    tool: ToolSpec,
    *,
    options: Optional[Mapping[str, Any]] = None,
    needs_prev_mesh: Optional[bool] = None,
) -> None:
    """Run one preprocessing *tool* in-process against *case*.

    Drives the tool's ``InitStep`` objects with a ``ctx`` seeded by
    :func:`seed_tool_ctx`. Any ``dict_file`` in *options* (or the tool's default) is
    absolutized against the case, so a relative dict path resolves without changing the
    working directory.

    *needs_prev_mesh* seeds ``ctx['_prev_mesh']`` from the on-disk mesh a prior
    ``block_mesh`` left — the input a refiner (snappy) or validator (checkMesh) reads.
    When ``None`` it defaults to the tool's ``consumes_mesh`` flag; a caller running a
    single tool against an existing mesh (the sweep runner) passes it explicitly.
    """
    opts: dict[str, Any] = dict(options or {})
    config_type = tool.step_config_type
    if config_type is not None and "dict_file" in config_type.model_fields:
        rel = opts.get("dict_file", config_type.model_fields["dict_file"].default)
        if not Path(rel).is_absolute():
            opts["dict_file"] = str(case.path / rel)
    if needs_prev_mesh is None:
        needs_prev_mesh = tool.consumes_mesh

    ctx = seed_tool_ctx(case, needs_prev_mesh=needs_prev_mesh)
    runtime = tool.instantiate({"tool": tool.name, **opts})
    for step in runtime.run_build():
        step.initializer(ctx)


def block_mesh(
    *, dict_file: str = "system/blockMeshDict", verbose: bool = False
) -> Step:
    """Generate the base mesh from a committed ``blockMeshDict`` (via ``blockMeshTool``)."""

    def step(case: CaseDir) -> None:
        run_tool(
            case, blockMeshTool, options={"dict_file": dict_file, "verbose": verbose}
        )

    return step


def snappy_hex_mesh(
    *,
    dict_file: str = "system/snappyHexMeshDict",
    overwrite: bool = True,
    verbose: bool = False,
) -> Step:
    """Refine the prior ``block_mesh`` output from ``snappyHexMeshDict`` (via ``snappyHexMeshTool``).

    Runs after a ``block_mesh`` step in the same pipeline; the background mesh it left on
    disk is reconstructed and fed to snappy as ``_prev_mesh``.
    """

    def step(case: CaseDir) -> None:
        run_tool(
            case,
            snappyHexMeshTool,
            options={
                "dict_file": dict_file,
                "overwrite": overwrite,
                "verbose": verbose,
            },
        )

    return step


class BoxMeshStep(BaseModel):
    """Parameters for a synthesized axis-aligned box ``blockMeshDict``."""

    n: tuple[int, int, int] = (3, 3, 1)
    dims: tuple[float, float, float] = (0.1, 0.1, 0.1)


def _render_box_blockmeshdict(cfg: BoxMeshStep) -> str:
    """Emit a ``blockMeshDict`` for an origin-anchored box (front/back ``empty`` when 2D)."""
    lx, ly, lz = cfg.dims
    nx, ny, nz = cfg.n
    fb_type = "empty" if nz == 1 else "wall"
    return f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}

convertToMeters 1;

vertices
(
    (0 0 0) ({lx} 0 0) ({lx} {ly} 0) (0 {ly} 0)
    (0 0 {lz}) ({lx} 0 {lz}) ({lx} {ly} {lz}) (0 {ly} {lz})
);

blocks ( hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1) );

edges ();

boundary
(
    walls {{ type wall; faces ((0 4 7 3)(2 6 5 1)(1 5 4 0)(3 7 6 2)); }}
    frontAndBack {{ type {fb_type}; faces ((0 3 2 1)(4 5 6 7)); }}
);

mergePatchPairs ();
"""


def box(
    *,
    n: tuple[int, int, int] = (3, 3, 1),
    dims: tuple[float, float, float] = (0.1, 0.1, 0.1),
    dict_file: str = "system/blockMeshDict",
) -> Step:
    """Synthesize a box ``blockMeshDict`` from *n*/*dims*, write it, then ``block_mesh`` it.

    Lets a test vary resolution by argument instead of committing dict variants. The case
    must still carry ``controlDict``/``fvSchemes``/``fvSolution`` (OpenFOAM mesh
    prerequisites) — as with :func:`block_mesh`.
    """
    cfg = BoxMeshStep(n=n, dims=dims)

    def step(case: CaseDir) -> None:
        target = case.path / dict_file
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(_render_box_blockmeshdict(cfg))
        run_tool(case, blockMeshTool, options={"dict_file": str(target)})

    return step
