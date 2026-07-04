# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Headless tests for the geometry AI helper (no network, no live LLM)."""

from __future__ import annotations

import pytest

pytest.importorskip("pybFoam")  # geometry_agent imports geometry → neofoam.tools

from neofoam.ui.geometry import PatchRole  # noqa: E402
from neofoam.ui.geometry_agent import (  # noqa: E402
    GeometryAssignments,
    RoleAssignment,
    apply_assignments,
)


def _rows() -> list[dict]:
    return [
        {
            "name": "tubes",
            "role": "wall",
            "is_snappy": True,
            "box_faces": None,
            "refinement_str": "1 2",
        },
        {
            "name": "inlet",
            "role": "inlet",
            "is_snappy": False,
            "box_faces": ["x_min"],
            "refinement_str": "",
        },
    ]


def test_apply_assignments_sets_role_and_surface_refinement():
    rows = _rows()
    assignments = GeometryAssignments(
        assignments=[
            RoleAssignment(patch="tubes", role=PatchRole.wall, refinement=(2, 3)),
        ]
    )
    out = apply_assignments(rows, assignments)

    tubes = next(r for r in out if r["name"] == "tubes")
    assert tubes["role"] == "wall"
    assert tubes["refinement"] == [2, 3]
    # Input rows are not mutated.
    assert "refinement" not in rows[0]


def test_apply_assignments_ignores_unknown_and_box_face_refinement():
    rows = _rows()
    assignments = GeometryAssignments(
        assignments=[
            RoleAssignment(patch="ghost", role=PatchRole.wall),  # unknown → ignored
            # refinement on a box-face patch is dropped (not a snappy surface).
            RoleAssignment(patch="inlet", role=PatchRole.outlet, refinement=(3, 4)),
        ]
    )
    out = apply_assignments(rows, assignments)

    inlet = next(r for r in out if r["name"] == "inlet")
    assert inlet["role"] == "outlet"
    assert "refinement" not in inlet
    assert {r["name"] for r in out} == {"tubes", "inlet"}


def test_empty_assignments_is_a_noop_copy():
    rows = _rows()
    out = apply_assignments(rows, GeometryAssignments())
    assert out == rows
    assert out is not rows
