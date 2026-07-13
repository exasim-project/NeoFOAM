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
    "SaveResultDTO",
]


class PatchDTO(BaseModel):
    """One boundary patch of a staged case (from its geometry manifest)."""

    name: str
    role: str
    stl: str | None = None


class CaseTextDTO(BaseModel):
    files: dict[str, str]


class CaseSpecDTO(BaseModel):
    values: dict[str, Any]


class SaveResultDTO(BaseModel):
    target_dir: str
    written: list[str]
    case_spec: dict[str, Any]
