# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Data models for graph validation diagnostics."""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple


@dataclass(frozen=True)
class GraphDiagnostic:
    """Machine-readable graph validation diagnostic."""

    code: Literal["duplicate_name", "missing_dependency", "cycle"]
    message: str
    node_name: Optional[str] = None
    dependency: Optional[str] = None
    cycle: Tuple[str, ...] = ()


@dataclass(frozen=True)
class GraphValidationReport:
    """Validation report for a dependency graph."""

    diagnostics: Tuple[GraphDiagnostic, ...]

    @property
    def is_valid(self) -> bool:
        return not self.diagnostics
