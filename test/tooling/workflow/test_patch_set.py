# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Stage-1 contract: the PatchSet schema (hermetic -- no FreeCAD/OpenFOAM)."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from neofoam.tooling.workflow.patch_set import (
    BoundingBox,
    PatchEntry,
    PatchSet,
    PatchRole,
)

FIXTURE = Path(__file__).parent / "cases" / "tube_bank_manifest.json"


def _manifest() -> PatchSet:
    return PatchSet(
        case_dir=".",
        geometry_source="tube_bank.FCStd",
        scale_to_meters=0.001,
        bbox=BoundingBox(min=(0.0, 0.0, 0.0), max=(0.576, 0.16, 0.02)),
        location_in_mesh=(0.08, 0.08, 0.01),
        length_scale=0.016,
        patches=[
            PatchEntry(
                name="inlet", stl="constant/triSurface/inlet.stl", role=PatchRole.inlet
            ),
            PatchEntry(
                name="outlet",
                stl="constant/triSurface/outlet.stl",
                role=PatchRole.outlet,
            ),
            PatchEntry(
                name="walls", stl="constant/triSurface/walls.stl", role=PatchRole.wall
            ),
            PatchEntry(
                name="tubes",
                stl="constant/triSurface/tubes.stl",
                role=PatchRole.wall,
                surface_refinement=(1, 2),
            ),
            PatchEntry(
                name="frontBack",
                stl="constant/triSurface/frontBack.stl",
                role=PatchRole.empty,
            ),
        ],
    )


def test_roundtrip_save_load(tmp_path: Path) -> None:
    """save() then load() reproduces the patch_set exactly."""
    patch_set = _manifest()
    path = patch_set.save(tmp_path / "manifest.json")
    assert PatchSet.load(path) == patch_set


def test_fixture_validates_and_matches_tube_bank() -> None:
    """The committed tube_bank fixture loads with the expected geometry + patches."""
    patch_set = PatchSet.load(FIXTURE)
    assert patch_set.geometry_source == "tube_bank.FCStd"
    assert patch_set.scale_to_meters == 0.001
    assert patch_set.bbox.max == (0.576, 0.16, 0.02)
    assert patch_set.patch_names() == ["inlet", "outlet", "walls", "tubes", "frontBack"]


def test_by_role_groups_both_walls() -> None:
    """Role queries drive boundary-condition selection in stage 3."""
    patch_set = PatchSet.load(FIXTURE)
    assert [p.name for p in patch_set.by_role(PatchRole.wall)] == ["walls", "tubes"]
    assert [p.name for p in patch_set.by_role(PatchRole.inlet)] == ["inlet"]
    assert [p.name for p in patch_set.by_role(PatchRole.empty)] == ["frontBack"]


def test_invalid_role_rejected() -> None:
    """Roles outside the enum are a validation error, not a silent passthrough."""
    with pytest.raises(ValidationError):
        PatchEntry(name="x", stl="x.stl", role="slipwall")  # type: ignore[arg-type]
