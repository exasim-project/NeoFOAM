# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Map ``validate_case`` findings to display rows for the Review step (pure)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

__all__ = ["FindingRow", "findings_to_rows", "save_error_rows"]

# Finding level → Vuetify VAlert `type`.
_LEVEL_COLOR = {"error": "error", "warning": "warning"}


@dataclass(frozen=True)
class FindingRow:
    """One validation finding, ready to render as a colored alert."""

    level: str
    color: str
    file: str
    message: str
    fix: str | None


def findings_to_rows(report: Any) -> list[FindingRow]:
    """Turn a ``ValidationReportDTO`` into display rows (level → alert color)."""
    return [
        FindingRow(
            level=f.level,
            color=_LEVEL_COLOR.get(f.level, "info"),
            file=f.file,
            message=f.message,
            fix=f.fix,
        )
        for f in report.findings
    ]


def _config_label(loc: tuple[Any, ...]) -> tuple[str, str]:
    """Split a pydantic error ``loc`` into (config name, dotted field path)."""
    parts = [str(p) for p in loc]
    if not parts:
        return "case_spec", ""
    return parts[0], ".".join(parts[1:])


def save_error_rows(exc: Exception) -> list[FindingRow]:
    """Turn a ``save_case`` failure into readable per-field rows.

    ``tools.save_case`` wraps pydantic's ``ValidationError`` in a ``ValueError``
    (``raise ... from exc``); recover the structured errors from ``__cause__`` (or from
    ``exc`` itself) and render one row per invalid field — e.g. ``boussinesq_config``
    / ``beta`` / *Field required* — instead of one wall of ``pydantic.dev`` URLs. Falls
    back to a single row when the cause is not a ``ValidationError``.
    """
    verr = exc if isinstance(exc, ValidationError) else exc.__cause__
    if isinstance(verr, ValidationError):
        rows = [
            FindingRow(
                level="error",
                color="error",
                file=config,
                message=f"{field}: {err['msg']}" if field else err["msg"],
                fix=None,
            )
            for err in verr.errors()
            for config, field in (_config_label(err.get("loc", ())),)
        ]
        if rows:
            return rows
    return [
        FindingRow(
            level="error",
            color="error",
            file="case_spec",
            message="Save failed — some configs are incomplete or invalid.",
            fix=str(exc),
        )
    ]
