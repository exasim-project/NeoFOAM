# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Pydantic v2 DTOs for the MCP introspection surface.

All fields are JSON-serializable — config classes are mapped to their
``__name__`` strings so no ``type`` object ever leaks into a tool response.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from neofoam.framework.validation import (
    Finding as FindingDTO,
    ValidationReport as ValidationReportDTO,
)
from neofoam.io.schema import (
    ConfigInfo as ConfigInfoDTO,
    ConfigSchema as ConfigSchemaDTO,
    ModelSummary as ModelEntryDTO,
    ToolInfo as ToolInfoDTO,
)

__all__ = [
    "ConfigInfoDTO",
    "ConfigSchemaDTO",
    "ModelEntryDTO",
    "ToolInfoDTO",
    "PatchDTO",
    "FindingDTO",
    "ValidationReportDTO",
    "CaseTextDTO",
    "CaseSpecDTO",
    "CaseSpecSchemaDTO",
    "ManifestSchemaDTO",
    "SaveResultDTO",
    "GeometryImportDTO",
    "LoadWarningDTO",
    "WorkspaceInfoDTO",
]


class PatchDTO(BaseModel):
    """One boundary patch of a staged case (from its geometry manifest)."""

    name: str
    role: str
    stl: str | None = None


class GeometryImportDTO(BaseModel):
    """Result of staging geometry into a case: the manifest written + its patches."""

    manifest: str
    """Path to the ``manifest.json`` written (the geometry hand-off contract)."""
    patches: list[PatchDTO]


class LoadWarningDTO(BaseModel):
    """A config file that was present on disk but could not be represented.

    Surfaces what :func:`load_case` silently dropped (F5): the config's file, its
    class name, and why it did not load — so an agent sees the gap instead of a
    clean object that quietly omits, e.g., a turbulence model we don't model yet.
    """

    file: str
    config: str
    reason: str


class CaseTextDTO(BaseModel):
    files: dict[str, str]


class CaseSpecDTO(BaseModel):
    values: dict[str, Any]
    warnings: list[LoadWarningDTO] = []
    """Configs present on disk but dropped (unrepresented/invalid) — see F5."""


class SaveResultDTO(BaseModel):
    target_dir: str
    written: list[str]
    case_spec: dict[str, Any]


class MeshInputsDTO(BaseModel):
    """Result of rendering the mesh dicts from a case's ``manifest.json``."""

    written: list[str]
    """The mesh-input files written (``system/blockMeshDict`` [+ ``snappyHexMeshDict``]
    + ``system/preprocess.yaml``)."""
    has_snappy: bool
    """True when a ``snappyHexMeshDict`` was produced (the manifest has a snappy
    surface); False for a pure blockMesh box."""


class ManifestSchemaDTO(BaseModel):
    """The geometry-manifest (``PatchSet``) JSON Schema, so ``import_geometry``'s
    ``manifest`` argument is self-describing (F1) — an agent authors it without
    reading ``workflow/patch_set.py``."""

    json_schema: dict[str, Any]
    required: list[str]
    """The manifest's required top-level keys (``geometry_source``, ``bbox`` …),
    hoisted out of ``json_schema`` so the contract is obvious at a glance."""


class CaseSpecSchemaDTO(BaseModel):
    """The aggregate ``save_case`` envelope (``CaseSpec``) JSON Schema, so the
    ``case_spec`` argument is self-describing — an agent learns the snake-case
    config keys without reading ``agent/case_fill.py``."""

    json_schema: dict[str, Any]
    config_keys: list[str]
    """The snake-case config field names the envelope accepts (all optional)."""


class WorkspaceInfoDTO(BaseModel):
    """The active path-confinement state, so a client learns the rules up front (F6)."""

    confined: bool
    """True when a workspace root confines filesystem tools (paths must be relative)."""
    root: str | None = None
    """The workspace root when confined, else ``None`` (absolute paths allowed)."""
