# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Case-free tool logic — plain ``f(solver, ...)`` functions returning DTOs.

These wrap the solver's case-free configuration seams
(:mod:`neofoam.framework.solver.configurations` + :mod:`neofoam.io`) and the
case read/load/save helpers into JSON-serializable Pydantic DTOs (the MCP surface
is agent-free — LLM case-fill is a wizard/library concern, not a tool here). They
take ``solver`` as a plain argument and import **no** ``fastmcp``/``fastapi`` — the
protocol layer in
:mod:`neofoam.mcp.server` wraps each one in a ``@mcp.tool`` decorator. Keeping
the logic here (not in closures) makes it unit-testable without the ``mcp`` extra.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from neofoam.agent.case_fill import (
    build_case_output_model,
    load_case_from_disk,
    read_case_text,
    save_case as _save_case,
)
from neofoam.framework.validation import validate
from neofoam.io.schema import (  # re-export: canonical home is neofoam.io.schema
    config_schema as config_schema,
    list_configs as list_configs,
    model_catalog as model_catalog,
    tool_catalog as tool_catalog,
)
from neofoam.mcp.dto import (
    CaseSpecDTO,
    CaseSpecSchemaDTO,
    CaseTextDTO,
    GeometryImportDTO,
    ManifestSchemaDTO,
    MeshInputsDTO,
    PatchDTO,
    SaveResultDTO,
    ValidationReportDTO,
    WorkspaceInfoDTO,
)
from neofoam.mcp.registry import list_solver_names
from neofoam.tooling import CaseAccessError, Workspace

INTROSPECTION_TOOL_NAMES: tuple[str, ...] = (
    "list_solvers",
    "model_catalog",
    "tool_catalog",
    "list_configs",
    "config_schema",
    "case_spec_schema",
    "manifest_schema",
)
WORKSPACE_TOOL_NAMES: tuple[str, ...] = ("workspace_info",)
GEOMETRY_TOOL_NAMES: tuple[str, ...] = (
    "import_geometry",
    "case_patches",
    "build_mesh_inputs",
)
VALIDATION_TOOL_NAMES: tuple[str, ...] = ("validate_case",)
SCAFFOLDING_TOOL_NAMES: tuple[str, ...] = ("read_case", "load_case", "save_case")
ALL_TOOL_NAMES: tuple[str, ...] = (
    INTROSPECTION_TOOL_NAMES
    + WORKSPACE_TOOL_NAMES
    + GEOMETRY_TOOL_NAMES
    + VALIDATION_TOOL_NAMES
    + SCAFFOLDING_TOOL_NAMES
)

# -- introspection (case-free) ------------------------------------------------


def list_solvers() -> list[str]:
    """Known solver spec names."""
    return list_solver_names()


def case_spec_schema(solver: Any) -> CaseSpecSchemaDTO:
    """JSON Schema of the ``save_case`` envelope (the aggregate ``CaseSpec``).

    Makes the ``case_spec`` argument self-describing: it lists every snake-case
    config key the envelope accepts (all optional — fill only the case's subset),
    so an agent authors ``save_case`` without reading ``agent/case_fill.py``.
    """
    model = build_case_output_model(solver=solver)
    schema = model.model_json_schema()
    return CaseSpecSchemaDTO(
        json_schema=schema,
        config_keys=sorted(model.model_fields),
    )


def manifest_schema() -> ManifestSchemaDTO:
    """JSON Schema of the geometry manifest (``PatchSet``) ``import_geometry`` writes.

    Makes the ``manifest`` argument self-describing — an agent learns the required
    ``geometry_source``/``bbox``/``location_in_mesh``/``length_scale`` keys and the
    per-patch shape without reading ``workflow/geometry.py``.
    """
    from neofoam.tooling.workflow.geometry import PatchSet

    schema = PatchSet.model_json_schema()
    required = list(schema.get("required", []))
    return ManifestSchemaDTO(json_schema=schema, required=required)


# -- workspace (path-confinement introspection) -------------------------------


def workspace_info(*, workspace: Workspace | None = None) -> WorkspaceInfoDTO:
    """Report whether path-confinement is on and the active root (F6).

    Without this an agent discovers the "paths must be relative under the root"
    rule only by triggering an opaque escape error. ``workspace is None`` means no
    root is configured (trusted local use — absolute paths allowed).
    """
    if workspace is None:
        return WorkspaceInfoDTO(confined=False, root=None)
    return WorkspaceInfoDTO(confined=True, root=str(workspace.root))


# -- case geometry ------------------------------------------------------------


def import_geometry(
    case_dir: str,
    manifest: dict[str, Any],
    *,
    stl_source_dir: str | None = None,
    workspace: Workspace | None = None,
) -> GeometryImportDTO:
    """Stage geometry into a case: write ``<case>/manifest.json`` (+ STLs), the
    hand-off contract :func:`case_patches` reads back (F1).

    ``manifest`` is a :class:`~neofoam.tooling.workflow.patch_set.PatchSet` dump — call
    :func:`manifest_schema` for the full JSON Schema. Its **required** keys are
    ``geometry_source`` (a provenance string, e.g. the CAD file or tool that produced
    the STLs), ``bbox`` (``{min, max}`` in metres), ``location_in_mesh`` (a point
    inside the fluid, metres) and ``length_scale`` (characteristic mesh length,
    metres); optional ``source_units`` (native CAD units, default ``"mm"``) and
    ``scale_to_meters`` (factor applied on export, default ``1.0``) record the
    coordinate system. ``patches`` is a list of
    ``{name, role, stl, box_faces?, surface_refinement?}``. ``case_dir`` is set from
    the resolved case, so the caller need not repeat it. For each patch the referenced
    STL must exist under ``<case>/<stl>`` — or, when ``stl_source_dir`` is given, its
    basename is copied from there into ``<case>/constant/triSurface/`` first (so a
    geometry producer that only emits STLs elsewhere can be bridged in one call).

    Paths are confined through ``workspace`` when given; a missing STL raises before
    the manifest is written, so a half-staged case is never left behind.
    """
    from neofoam.tooling.workflow.geometry import PatchSet

    case = _confine(case_dir, workspace)
    case.mkdir(parents=True, exist_ok=True)
    tri_dir = case / "constant" / "triSurface"

    source = (
        _confine_existing(stl_source_dir, workspace, kind="stl_source_dir")
        if stl_source_dir is not None
        else None
    )

    patch_set = PatchSet.model_validate({**manifest, "case_dir": str(case)})

    for patch in patch_set.patches:
        rel = Path(patch.stl)
        if source is not None:
            tri_dir.mkdir(parents=True, exist_ok=True)
            src = source / rel.name
            if not src.is_file():
                raise ValueError(f"STL for patch {patch.name!r} not found: {src}")
            staged = tri_dir / rel.name
            shutil.copy2(src, staged)
            patch.stl = str(staged.relative_to(case))
        elif not (case / rel).is_file():
            raise ValueError(
                f"STL for patch {patch.name!r} not found under case: {case / rel} "
                "(stage the STLs first, or pass stl_source_dir)"
            )

    written = patch_set.save(case / "manifest.json")
    return GeometryImportDTO(
        manifest=str(written),
        patches=[
            PatchDTO(name=p.name, role=p.role.value, stl=p.stl)
            for p in patch_set.patches
        ],
    )


def case_patches(
    case_dir: str, *, workspace: Workspace | None = None
) -> list[PatchDTO]:
    """Boundary patches (name + role) of a staged case, from ``<case>/manifest.json``.

    Lets an agent author boundary conditions without inventing patch names or roles:
    the geometry is a given, extracted upstream and written to the manifest.
    ``case_dir`` is confined through ``workspace`` (when given) and must exist.
    """
    from neofoam.tooling.workflow.geometry import PatchSet

    case = _confine_existing(case_dir, workspace)
    manifest = Path(case) / "manifest.json"
    if not manifest.is_file():
        raise ValueError(
            f"no geometry manifest at {manifest} (stage the geometry first)"
        )
    patch_set = PatchSet.load(manifest)
    return [
        PatchDTO(name=p.name, role=p.role.value, stl=p.stl) for p in patch_set.patches
    ]


def build_mesh_inputs(
    case_dir: str, *, workspace: Workspace | None = None
) -> MeshInputsDTO:
    """Render the mesh dicts for a staged case from its ``<case>/manifest.json``.

    Closes the manifest → mesh gap: reads the :class:`~neofoam.tooling.workflow.patch_set.PatchSet`
    and writes ``system/blockMeshDict`` (background box + the ``box_faces`` boundary
    patches), a ``system/snappyHexMeshDict`` **only when** a patch is a snappy surface,
    and a ``system/preprocess.yaml`` enable-list (blockMesh → [snappyHexMesh] →
    checkMesh) — so ``neofoam preprocess`` can build the mesh with no hand-authored
    dicts. ``case_dir`` is confined through ``workspace`` (when given) and must contain
    a manifest.
    """
    from neofoam.io import write_configs
    from neofoam.tooling.workflow.geometry import PatchSet
    from neofoam.tooling.workflow.geometry import build_mesh_inputs as _build

    case = _confine_existing(case_dir, workspace)
    manifest = Path(case) / "manifest.json"
    if not manifest.is_file():
        raise ValueError(
            f"no geometry manifest at {manifest} (stage the geometry first)"
        )
    patch_set = PatchSet.load(manifest)
    block, snappy, pre = _build(patch_set)

    configs = [block, pre] + ([snappy] if snappy is not None else [])
    report = write_configs(configs, case_dir=str(case))
    return MeshInputsDTO(
        written=sorted(report.keys()),
        has_snappy=snappy is not None,
    )


# -- case validation (static pre-flight) --------------------------------------


def validate_case(
    solver: Any, case_dir: str, *, workspace: Workspace | None = None
) -> ValidationReportDTO:
    """Static pre-flight of an authored case — no OpenFOAM run.

    Delegates to :func:`neofoam.framework.validation.validate` (the CheckRegistry of
    isolated checks). Kept here as a thin frontend shim so the MCP tool surface and
    the existing tests keep working; the check logic lives in ``framework.validation``.
    ``case_dir`` is confined through ``workspace`` (when given) and must exist.
    """
    case = _confine_existing(case_dir, workspace)
    return validate(solver, str(case))


# -- case scaffolding ---------------------------------------------------------


def _confine(path: str, workspace: Workspace | None) -> Path:
    """Confine ``path`` via ``workspace`` (a write target — need not exist).

    ``None`` keeps the legacy trusted-absolute behavior (today's direct callers).
    """
    if workspace is not None:
        return workspace.resolve(path)
    return Path(path)


def _confine_existing(
    path: str, workspace: Workspace | None, *, kind: str = "case_dir"
) -> Path:
    """Confine ``path`` and require it to be an existing, readable directory.

    ``kind`` labels the path in the rejection message so an escaping/absent
    ``source_dir`` is distinguishable from a bad ``case_dir`` under a workspace too.
    """
    if workspace is not None:
        return workspace.resolve_existing(path, kind=kind)
    resolved = Path(path)
    if not resolved.is_dir():
        raise CaseAccessError(f"{kind} does not exist or is not a directory: {path!r}")
    return resolved


def read_case(
    solver: Any, case_dir: str, *, workspace: Workspace | None = None
) -> CaseTextDTO:
    """Read every config-bound file of a case as raw text.

    ``case_dir`` is confined through ``workspace`` (when given) and must be an
    existing, readable directory.
    """
    case = _confine_existing(case_dir, workspace)
    return CaseTextDTO(files=read_case_text(str(case), solver=solver))


def load_case(
    solver: Any, case_dir: str, *, workspace: Workspace | None = None
) -> CaseSpecDTO:
    """Load each present config from disk into an aggregate CaseSpec dump.

    ``case_dir`` is confined through ``workspace`` (when given) and must be an
    existing, readable directory.
    """
    case = _confine_existing(case_dir, workspace)
    warnings: list[dict[str, str]] = []
    spec = load_case_from_disk(str(case), solver=solver, warnings=warnings)
    return CaseSpecDTO(values=spec.model_dump(), warnings=warnings)


def save_case(
    solver: Any,
    case_spec: dict[str, Any],
    target_dir: str,
    *,
    workspace: Workspace | None = None,
) -> SaveResultDTO:
    """Validate ``case_spec`` against the solver's aggregate model, then write.

    ``target_dir`` is confined through ``workspace`` (when given) before any write, so
    a malformed *or* escaping payload leaves the filesystem untouched.
    """
    target = _confine(target_dir, workspace)
    model = build_case_output_model(solver=solver)
    try:
        validated = model(**case_spec)
    except ValidationError as exc:
        raise ValueError(f"invalid case_spec for solver: {exc}") from exc
    written = _save_case(validated, target)
    return SaveResultDTO(
        target_dir=str(target),
        written=[str(p) for p in written],
        case_spec=validated.model_dump(),
    )
