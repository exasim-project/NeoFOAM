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
    CaseTextDTO,
    PatchDTO,
    SaveResultDTO,
    ValidationReportDTO,
)
from neofoam.mcp.registry import list_solver_names
from neofoam.tooling import CaseAccessError, Workspace

INTROSPECTION_TOOL_NAMES: tuple[str, ...] = (
    "list_solvers",
    "model_catalog",
    "tool_catalog",
    "list_configs",
    "config_schema",
)
GEOMETRY_TOOL_NAMES: tuple[str, ...] = ("case_patches",)
VALIDATION_TOOL_NAMES: tuple[str, ...] = ("validate_case",)
SCAFFOLDING_TOOL_NAMES: tuple[str, ...] = ("read_case", "load_case", "save_case")
ALL_TOOL_NAMES: tuple[str, ...] = (
    INTROSPECTION_TOOL_NAMES
    + GEOMETRY_TOOL_NAMES
    + VALIDATION_TOOL_NAMES
    + SCAFFOLDING_TOOL_NAMES
)

# -- introspection (case-free) ------------------------------------------------


def list_solvers() -> list[str]:
    """Known solver spec names."""
    return list_solver_names()


# -- case geometry ------------------------------------------------------------


def case_patches(
    case_dir: str, *, workspace: Workspace | None = None
) -> list[PatchDTO]:
    """Boundary patches (name + role) of a staged case, from ``<case>/manifest.json``.

    Lets an agent author boundary conditions without inventing patch names or roles:
    the geometry is a given, extracted upstream and written to the manifest.
    ``case_dir`` is confined through ``workspace`` (when given) and must exist.
    """
    from neofoam.workflow.patch_set import PatchSet

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
    spec = load_case_from_disk(str(case), solver=solver)
    return CaseSpecDTO(values=spec.model_dump())


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
