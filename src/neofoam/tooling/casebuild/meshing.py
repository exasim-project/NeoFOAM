# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Injectable meshing strategies as pipeline steps: ``block_mesh``, ``box``, ``snappy_hex_mesh``.

There is deliberately no default meshing verb — a case's mesh is chosen by injecting
one of these steps. Each reuses the shared ``neofoam.tools`` mesh tools
(``blockMeshTool`` / ``snappyHexMeshTool`` + their pydantic step configs) rather than
re-wrapping ``pybFoam.meshing``: :func:`_run_tool` drives a tool's ``InitStep`` objects
directly against a :class:`CaseDir`, seeding the ``ctx`` the tool expects
(``_foam_time``, and ``_prev_mesh`` for snappy). A tool's ``dict_file`` is passed as an
absolute path so the Foam ``Time`` can use the two-arg (root, case) form — no ``chdir``
and no ``argList`` lifetime hazard.

``block_mesh`` reads a committed ``blockMeshDict``; ``box`` synthesizes one from
``n``/``dims`` (the OpenFOAM ``blocks``/``boundary`` grammar can't be built through the
typed ``dictionary`` API, so it is emitted as text); ``snappy_hex_mesh`` refines the mesh
a prior ``block_mesh`` left on disk (reconstructed via ``pyf.fvMesh``).
"""

from __future__ import annotations

from typing import Any

import pybFoam as pyf
from pydantic import BaseModel

from neofoam.tooling.casebuild.pipeline import CaseDir, Step
from neofoam.tools.block_mesh import blockMeshTool
from neofoam.tools.snappy_hex_mesh import snappyHexMeshTool


def _run_tool(
    case: CaseDir,
    tool: Any,
    entry: dict[str, Any],
    *,
    needs_prev_mesh: bool = False,
) -> None:
    """Drive *tool*'s InitSteps against *case* with a seeded ``ctx``.

    Builds the Foam ``Time`` with the two-arg (root, case) form so no ``chdir`` is
    needed (the tool's ``dict_file`` in *entry* must be absolute). When
    *needs_prev_mesh*, reconstructs the on-disk mesh (left by a prior ``block_mesh``)
    as ``ctx['_prev_mesh']`` — the input snappy refines.
    """
    time = pyf.Time(str(case.path.parent), case.path.name)
    ctx: dict[str, Any] = {"_foam_time": time}
    if needs_prev_mesh:
        ctx["_prev_mesh"] = pyf.fvMesh(time)
    runtime = tool.instantiate(entry)
    for step in runtime.run_build():
        step.initializer(ctx)


def block_mesh(
    *, dict_file: str = "system/blockMeshDict", verbose: bool = False
) -> Step:
    """Generate the base mesh from a committed ``blockMeshDict`` (via ``blockMeshTool``)."""

    def step(case: CaseDir) -> None:
        entry = {
            "tool": "blockMesh",
            "dict_file": str(case.path / dict_file),
            "verbose": verbose,
        }
        _run_tool(case, blockMeshTool, entry)

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
        entry = {
            "tool": "snappyHexMesh",
            "dict_file": str(case.path / dict_file),
            "overwrite": overwrite,
            "verbose": verbose,
        }
        _run_tool(case, snappyHexMeshTool, entry, needs_prev_mesh=True)

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
        entry = {"tool": "blockMesh", "dict_file": str(target), "verbose": False}
        _run_tool(case, blockMeshTool, entry)

    return step
