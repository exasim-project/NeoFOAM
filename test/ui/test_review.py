# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for the validate_case → display-row mapping."""

from __future__ import annotations

from pydantic import BaseModel, ValidationError

from neofoam.mcp.dto import FindingDTO, ValidationReportDTO
from neofoam.ui.review import findings_to_rows, save_error_rows


def test_findings_map_level_to_color_and_preserve_fields():
    report = ValidationReportDTO(
        ok=False,
        findings=[
            FindingDTO(
                level="error",
                file="constant/g",
                message="Boussinesq enabled but gravity is missing",
                fix="add constant/g with value (0 -9.81 0)",
            ),
            FindingDTO(
                level="warning",
                file="system/fvSolution",
                message="GAMG solver for p_rgh has no smoother",
                fix=None,
            ),
        ],
    )
    rows = findings_to_rows(report)

    assert [r.color for r in rows] == ["error", "warning"]
    assert rows[0].file == "constant/g"
    assert rows[0].fix and "constant/g" in rows[0].fix
    assert rows[1].fix is None


def test_empty_report_yields_no_rows():
    assert findings_to_rows(ValidationReportDTO(ok=True, findings=[])) == []


class _Boussinesq(BaseModel):
    beta: float
    TRef: float


def test_save_error_rows_split_validation_error_into_per_field_rows():
    # Reproduce tools.save_case's wrapping: `raise ValueError(...) from validation_error`.
    try:
        _Boussinesq()  # type: ignore[call-arg]  # both fields missing
    except ValidationError as verr:
        exc: Exception = ValueError("invalid case_spec for solver: ...")
        exc.__cause__ = verr

    rows = save_error_rows(exc)
    assert len(rows) == 2
    assert {r.file for r in rows} == {"beta", "TRef"}
    assert all("Field required" in r.message for r in rows)
    assert all(r.color == "error" for r in rows)


def test_save_error_rows_fallback_when_not_a_validation_error():
    rows = save_error_rows(RuntimeError("disk full"))
    assert len(rows) == 1
    assert rows[0].file == "case_spec"
    assert rows[0].fix == "disk full"
