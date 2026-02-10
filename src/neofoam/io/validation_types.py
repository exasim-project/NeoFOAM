# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""Shared validation types to avoid circular imports."""

from dataclasses import dataclass
from typing import Any, Optional, Type

from pydantic import BaseModel


@dataclass(frozen=True)
class ValidationErrors:
    field: Any
    error_type: str
    message: str
    file_name: str
    input_value: Any = None
    subdict: Optional[str] = None


@dataclass(frozen=True)
class ModelInputDefinition:
    baseModel: Type[BaseModel]
    relative_path: str
    encoding: str = "utf-8"
    required: bool = True
    description: str = ""
