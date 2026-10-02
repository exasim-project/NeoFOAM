# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""Shared validation types to avoid circular imports."""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class IOMetadata:
    """Metadata set by the ``@IOStrategy`` decorator.

    ``reader``/``writer`` are format-marker strategy instances (``OpenFOAMStrategy``,
    ``YAMLStrategy``, ``JSONStrategy``): reads/writes go through
    :class:`neofoam.io.DictFile`, so these are used only for their declared
    ``subdict_path`` and, in :func:`neofoam.io.write_configs`, their type.
    """

    file: str
    reader: Any
    writer: Any

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
