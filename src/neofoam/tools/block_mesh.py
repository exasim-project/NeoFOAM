# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""blockMesh tool — the build step *and* the ``system/blockMeshDict`` writer config.

Two things live here, both keyed to the ``blockMesh`` tool:

* :class:`BlockMeshStep` / :data:`blockMeshTool` — the in-process ``@build``
  initializer (generate the base mesh from ``blockMeshDict``). The initializer
  resolves the pybFoam ``Time`` from the live ``ctx`` at call time (never captured
  in a long-lived closure — that would create a mesh-bound reference cycle that
  segfaults across in-process runs). ``pyf`` / ``generate_blockmesh`` are module
  globals so tests can monkeypatch them without building a mesh.

* :class:`BlockMeshDictConfig` — a ``BaseConfig`` bound to ``system/blockMeshDict``
  that both **writes** the file (so a case can be scaffolded from a filled model,
  like every other NeoFOAM config) and **reads** one back (so an existing dict
  round-trips). The pydantic shape and the on-disk shape deliberately differ: the
  OpenFOAM writer serialises only ``dict`` + scalars, so the compound sections
  (``vertices`` / ``blocks`` / ``boundary``) are emitted as OpenFOAM-syntax
  **strings** via a ``model_serializer`` gated on ``context={"format":
  "openfoam"}`` (the :class:`~neofoam.fields.value_types.FieldValue` trick), and
  parsed back by a ``model_validator``. Exotic parts (multi/edge grading, curved
  ``edges``, projected ``faces``, the legacy ``patches`` form, grading variables)
  are preserved as opaque token passthroughs so arbitrary dicts round-trip.
"""

from typing import Any, Literal, Optional, Union

import pybFoam as pyf
from pybFoam.meshing import generate_blockmesh
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializationInfo,
    model_serializer,
    model_validator,
)

from neofoam.framework.initialization import InitStep, lazy
from neofoam.framework.tools import Tool
from neofoam.io import OF, BaseConfig, IOStrategy

from ._foam_tokens import point, read_group, tokenize
from .registry import register_tool

# --------------------------------------------------------------------------- #
# The build step                                                              #
# --------------------------------------------------------------------------- #


class BlockMeshStep(BaseModel):
    """Generate the base mesh in-process from ``blockMeshDict``."""

    tool: Literal["blockMesh"]
    dict_file: str = "system/blockMeshDict"
    verbose: bool = False


blockMeshTool = Tool("blockMesh")


@blockMeshTool.build
def _build_block(cfg: BlockMeshStep) -> list[InitStep]:
    def gen(ctx: dict[str, Any]) -> Any:
        return generate_blockmesh(
            ctx["_foam_time"],
            pyf.dictionary.read(cfg.dict_file),
            verbose=cfg.verbose,
        )

    return [lazy("preprocess.blockMesh", gen, depends_on=["_foam_time"])]


register_tool(blockMeshTool)


# --------------------------------------------------------------------------- #
# The blockMeshDict writer/reader config                                      #
# --------------------------------------------------------------------------- #

Vec3 = tuple[float, float, float]


class Block(BaseModel):
    """One ``hex`` block: 8 vertex indices, cell counts, grading, optional zone."""

    vertices: list[int]
    """The 8 corner vertex indices (blockMesh hex order)."""
    cells: tuple[int, int, int]
    """``(nx, ny, nz)`` cell counts."""
    grading: str = "simpleGrading ( 1 1 1 )"
    """Full grading spec (opaque) — ``simpleGrading (...)`` / ``edgeGrading (...)``,
    including multi-grading sublists — preserved verbatim for faithful round-trips."""
    zone: Optional[str] = None
    """Optional cellZone name placed between the vertex and cell groups."""


class BlockPatch(BaseModel):
    """One ``boundary`` entry: a named patch owning quad faces (+ opaque extras)."""

    name: str
    type: str = "patch"
    faces: list[tuple[int, int, int, int]] = Field(default_factory=list)
    extra: dict[str, str] = Field(default_factory=dict)
    """Any other patch keys (``inGroups`` / ``neighbourPatch`` / ...), value =
    normalised token string, preserved for round-trip fidelity."""


@IOStrategy(OF("system/blockMeshDict"))
class BlockMeshDictConfig(BaseConfig):
    """``system/blockMeshDict`` — a general single-file blockMesh representation.

    Models the common structure (``scale`` + ``vertices`` + ``blocks`` topology +
    ``boundary`` patches) and preserves everything else (curved ``edges``,
    projected ``faces``, the legacy ``patches`` form, ``defaultPatch``, grading
    variables, ...) as opaque passthroughs, so it can load, write and reproduce
    arbitrary tutorial dicts. Construct one directly (or via
    :func:`neofoam.tooling.workflow.mesh_inputs.block_mesh_dict`) to scaffold a case.

    ``vertices`` / ``blocks`` / ``boundary`` are ``list | str``: normally the
    structured list, but a section that does not fit the structured grammar
    (projected/named vertices, expression-valued coordinates, ...) is kept as the
    raw OpenFOAM string and emitted verbatim — so a dict always round-trips even
    when it cannot be fully structured.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    scale: float = 1.0
    vertices: Union[list[Vec3], str] = Field(default_factory=list)
    blocks: Union[list[Block], str] = Field(default_factory=list)
    edges: str = "( )"
    boundary: Union[list[BlockPatch], str] = Field(default_factory=list)
    merge_patch_pairs: str = Field(default="( )", alias="mergePatchPairs")

    @model_validator(mode="before")
    @classmethod
    def _parse_foam(cls, data: Any) -> Any:
        """Parse the OpenFOAM token strings (from ``load``) into structured fields.

        Only strings are parsed, so a directly-constructed / JSON model (already
        structured) passes through untouched. A section that does not fit the
        structured grammar keeps its raw string (the ``str`` arm of the field),
        which the serializer emits verbatim. ``FoamFile`` is dropped — the writer
        re-injects a fresh header.
        """
        if not isinstance(data, dict):
            return data
        data = dict(data)
        data.pop("FoamFile", None)
        for key, parser in (
            ("vertices", _parse_vertices),
            ("blocks", _parse_blocks),
            ("boundary", _parse_boundary),
        ):
            value = data.get(key)
            if isinstance(value, str):
                try:
                    data[key] = parser(value)
                except Exception:
                    pass  # keep the raw string; it round-trips verbatim
        return data

    @model_serializer(mode="wrap")
    def _serialize(self, handler: Any, info: SerializationInfo) -> dict[str, Any]:
        """Structured data for JSON/YAML/forms; OpenFOAM text under ``openfoam``."""
        if (info.context or {}).get("format") != "openfoam":
            return handler(self)  # type: ignore[no-any-return]
        vertices = self.vertices
        blocks = self.blocks
        out: dict[str, Any] = {
            "scale": self.scale,
            "vertices": vertices
            if isinstance(vertices, str)
            else "( " + " ".join(point(v) for v in vertices) + " )",
            "blocks": blocks
            if isinstance(blocks, str)
            else "( " + " ".join(_fmt_block(b) for b in blocks) + " )",
            "edges": self.edges,
        }
        # Only emit ``boundary`` when populated: a dict using the legacy ``patches``
        # form (preserved as an extra) must not gain a spurious empty ``boundary``.
        boundary = self.boundary
        if isinstance(boundary, str):
            out["boundary"] = boundary
        elif boundary:
            out["boundary"] = "( " + " ".join(_fmt_patch(p) for p in boundary) + " )"
        out["mergePatchPairs"] = self.merge_patch_pairs
        # Preserve any unmodeled top-level entries (patches / faces / defaultPatch /
        # grading variables / ...) so an arbitrary dict reproduces faithfully.
        for key, value in (self.__pydantic_extra__ or {}).items():
            out[key] = value
        return out


# -- parsers (OpenFOAM token string → structured) --------------------------- #


def _parse_vertices(text: str) -> list[Vec3]:
    inner, _ = read_group(tokenize(text), 0)
    nums = [float(t) for t in inner if t not in "()"]
    return [(nums[i], nums[i + 1], nums[i + 2]) for i in range(0, len(nums), 3)]


def _parse_blocks(text: str) -> list[Block]:
    inner, _ = read_group(tokenize(text), 0)
    blocks: list[Block] = []
    i = 0
    while i < len(inner):
        if inner[i] != "hex":
            i += 1  # defensively skip anything unexpected
            continue
        i += 1
        verts, i = read_group(inner, i)
        zone = None
        if i < len(inner) and inner[i] != "(":
            zone = inner[i]
            i += 1
        cells, i = read_group(inner, i)
        if i < len(inner) and inner[i] in ("simpleGrading", "edgeGrading"):
            gtype = inner[i]
            grad, i = read_group(inner, i + 1)
            grading = f"{gtype} ( {' '.join(grad)} )"
        else:
            grading = "simpleGrading ( 1 1 1 )"
        blocks.append(
            Block(
                vertices=[int(v) for v in verts],
                zone=zone,
                cells=(int(cells[0]), int(cells[1]), int(cells[2])),
                grading=grading,
            )
        )
    return blocks


def _parse_boundary(text: str) -> list[BlockPatch]:
    inner, _ = read_group(tokenize(text), 0)
    patches: list[BlockPatch] = []
    i = 0
    while i < len(inner):
        name = inner[i]
        i += 1
        body, i = read_group(inner, i, open_="{", close="}")
        patches.append(_parse_patch(name, body))
    return patches


def _parse_patch(name: str, body: list[str]) -> BlockPatch:
    ptype = "patch"
    faces: list[tuple[int, int, int, int]] = []
    extra: dict[str, str] = {}
    j = 0
    while j < len(body):
        key = body[j]
        j += 1
        value: list[str] = []
        depth = 0
        while j < len(body):
            tok = body[j]
            j += 1
            if tok == ";" and depth == 0:
                break
            if tok in "({":
                depth += 1
            elif tok in ")}":
                depth -= 1
            value.append(tok)
        if key == "type":
            ptype = value[0] if value else "patch"
        elif key == "faces":
            faces = _parse_faces(value)
        else:
            extra[key] = " ".join(value)
    return BlockPatch(name=name, type=ptype, faces=faces, extra=extra)


def _parse_faces(tokens: list[str]) -> list[tuple[int, int, int, int]]:
    nums = [int(t) for t in tokens if t not in "()"]
    return [
        (nums[i], nums[i + 1], nums[i + 2], nums[i + 3]) for i in range(0, len(nums), 4)
    ]


# -- formatters (structured → OpenFOAM token string) ------------------------ #


def _fmt_block(b: Block) -> str:
    verts = " ".join(str(v) for v in b.vertices)
    zone = f" {b.zone}" if b.zone else ""
    nx, ny, nz = b.cells
    return f"hex ( {verts} ){zone} ( {nx} {ny} {nz} ) {b.grading}"


def _fmt_patch(p: BlockPatch) -> str:
    faces = " ".join(f"( {a} {b} {c} {d} )" for (a, b, c, d) in p.faces)
    body = f"type {p.type} ; faces ( {faces} ) ;"
    for key, value in p.extra.items():
        body += f" {key} {value} ;"
    return f"{p.name} {{ {body} }}"
