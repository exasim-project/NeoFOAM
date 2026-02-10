# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""Protocols for IO reading and writing strategies."""

from typing import Any, Protocol
from pathlib import Path


class ReadingStrategy(Protocol):
    """Protocol for configuration reading strategies."""

    def read(self, path: Path, encoding: str = "utf-8") -> dict[str, Any]:
        """Read and parse configuration file."""
        ...


class WritingStrategy(Protocol):
    """Protocol for configuration writing strategies."""

    def write(self, data: dict[str, Any], path: Path, encoding: str = "utf-8") -> None:
        """Write configuration to file."""
        ...
