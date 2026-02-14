# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""Shared validation types to avoid circular imports."""

from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable


@runtime_checkable
class ReadingStrategy(Protocol):
    """Protocol for reading configuration files."""

    def read(self, path: Any, encoding: str = "utf-8") -> dict[str, Any]: ...


@runtime_checkable
class WritingStrategy(Protocol):
    """Protocol for writing configuration files."""

    def write(
        self, data: dict[str, Any], path: Any, encoding: str = "utf-8"
    ) -> None: ...


@dataclass
class IOMetadata:
    """Metadata set by the ``@IOStrategy`` decorator."""

    file: str
    reader: ReadingStrategy
    writer: WritingStrategy

    @property
    def subdict(self) -> Optional[str]:
        """Get the subdict path from the reader, if any."""
        return getattr(self.reader, "subdict_path", None)


@dataclass(frozen=True)
class ValidationErrors:
    field: Any
    error_type: str
    message: str
    file_name: str
    input_value: Any = None
    subdict: Optional[str] = None
