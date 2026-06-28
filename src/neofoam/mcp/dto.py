# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Pydantic v2 DTOs for the MCP introspection surface.

All fields are JSON-serializable — config classes are mapped to their
``__name__`` strings so no ``type`` object ever leaks into a tool response.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ConfigInfoDTO(BaseModel):
    name: str
    cls_name: str
    file: str | None = None


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


class ToggleModelDTO(BaseModel):
    name: str
    label: str
    dicts: list[str]
    fields: list[str]

    @classmethod
    def from_toggle(cls, toggle: Any) -> "ToggleModelDTO":
        return cls(
            name=toggle.name,
            label=toggle.label,
            dicts=[c.__name__ for c in toggle.dicts],
            fields=[c.__name__ for c in toggle.fields],
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
