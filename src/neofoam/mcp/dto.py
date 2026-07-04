# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Pydantic v2 DTOs for the MCP introspection surface.

All fields are JSON-serializable — config classes are mapped to their
``__name__`` strings so no ``type`` object ever leaks into a tool response.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class ConfigInfoDTO(BaseModel):
    name: str
    cls_name: str
    file: str | None = None
    description: str | None = None


class PatchDTO(BaseModel):
    """One boundary patch of a staged case (from its geometry manifest)."""

    name: str
    role: str
    stl: str | None = None


class FindingDTO(BaseModel):
    """One issue found by ``validate_case`` — with a concrete fix where possible."""

    level: Literal["error", "warning"]
    file: str
    message: str
    fix: str | None = None


class ValidationReportDTO(BaseModel):
    """The result of a static pre-flight check of a case (no OpenFOAM run)."""

    ok: bool
    findings: list[FindingDTO]


class ToolInfoDTO(BaseModel):
    """One registered preprocessing tool + the config it reads.

    ``name`` is what goes in a ``PreprocessConfig`` ``tools`` entry's ``tool`` key
    (``blockMesh`` / ``snappyHexMesh`` / ``checkMesh``); ``step_schema`` is the JSON
    Schema of that entry (its options + ``depends_on``). ``dict_file`` is the case
    file the tool reads at run time and ``config`` the config class that writes it
    (``blockMesh`` → ``BlockMeshDictConfig`` at ``system/blockMeshDict``); both are
    ``None`` for a tool that reads no dict (e.g. ``checkMesh``).
    """

    name: str
    description: str | None = None
    step_schema: dict[str, Any]
    dict_file: str | None = None
    config: str | None = None


class ModelEntryDTO(BaseModel):
    name: str
    label: str
    required: bool
    dicts: list[str]
    fields: list[str]

    @classmethod
    def from_entry(cls, entry: Any) -> "ModelEntryDTO":
        return cls(
            name=entry.name,
            label=entry.label,
            required=entry.required,
            dicts=[c.__name__ for c in entry.dicts],
            fields=[c.__name__ for c in entry.fields],
        )


class ConfigSchemaDTO(BaseModel):
    name: str
    json_schema: dict[str, Any]
    ui_schema: dict[str, Any]
    defaults: dict[str, Any]


class CaseTextDTO(BaseModel):
    files: dict[str, str]


class CaseSpecDTO(BaseModel):
    values: dict[str, Any]


class SaveResultDTO(BaseModel):
    target_dir: str
    written: list[str]
    case_spec: dict[str, Any]
