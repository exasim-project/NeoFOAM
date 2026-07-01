# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Stage-1 integration: extract STL patches from tube_bank.FCStd (FreeCAD-gated).

Needs the FreeCAD/foamcadagent conda env. Point ``NEOFOAM_CAD_PYTHON`` at that
env's interpreter (and optionally ``NEOFOAM_TUBE_BANK`` at the model); otherwise
the test is skipped. The interpreter runs the driver directly (``python_bin=``),
so conda need not be on PATH.
"""

import os
from pathlib import Path

import pytest

from neofoam.e2e.extract import extract_stl
from neofoam.e2e.manifest import PatchRole

_CAD_PYTHON_CANDIDATES = [
    os.environ.get("NEOFOAM_CAD_PYTHON"),
    str(Path.home() / "miniforge3" / "envs" / "foamcadagent" / "bin" / "python"),
    str(Path.home() / "miniconda3" / "envs" / "foamcadagent" / "bin" / "python"),
]
_TUBE_BANK_CANDIDATES = [
    os.environ.get("NEOFOAM_TUBE_BANK"),
    str(Path.home() / "libsAndApps" / "FoamCADAgent" / "examples" / "tube_bank.FCStd"),
]


def _first_existing(candidates: list[str | None]) -> str | None:
    return next((c for c in candidates if c and Path(c).is_file()), None)


CAD_PYTHON = _first_existing(_CAD_PYTHON_CANDIDATES)
TUBE_BANK = _first_existing(_TUBE_BANK_CANDIDATES)

pytestmark = pytest.mark.skipif(
    not (CAD_PYTHON and TUBE_BANK),
    reason="FreeCAD/foamcadagent env or tube_bank.FCStd not available "
    "(set NEOFOAM_CAD_PYTHON / NEOFOAM_TUBE_BANK)",
)


def test_extract_tube_bank(tmp_path: Path) -> None:
    """Extraction writes 5 role-tagged STL patches + a valid manifest."""
    manifest = extract_stl(TUBE_BANK, tmp_path, python_bin=CAD_PYTHON)

    # Patches: the expected set, in deterministic order, with correct roles.
    assert manifest.patch_names() == ["inlet", "outlet", "walls", "tubes", "frontBack"]
    roles = {p.name: p.role for p in manifest.patches}
    assert roles == {
        "inlet": PatchRole.inlet,
        "outlet": PatchRole.outlet,
        "walls": PatchRole.wall,
        "tubes": PatchRole.wall,
        "frontBack": PatchRole.empty,
    }

    # Every patch's STL exists, is non-empty, and is a real ASCII solid.
    for patch in manifest.patches:
        stl = tmp_path / patch.stl
        assert stl.is_file() and stl.stat().st_size > 0
        assert "facet normal" in stl.read_text()

    # Geometry is in metres and the seed point sits strictly inside the bbox.
    assert manifest.scale_to_meters == 0.001
    assert manifest.bbox.max == pytest.approx((0.576, 0.16, 0.02))
    lo, hi, pt = manifest.bbox.min, manifest.bbox.max, manifest.location_in_mesh
    assert all(lo[i] < pt[i] < hi[i] for i in range(3))


def test_missing_model_raises(tmp_path: Path) -> None:
    """A non-existent CAD model fails fast before spawning the driver."""
    with pytest.raises(FileNotFoundError):
        extract_stl(tmp_path / "nope.FCStd", tmp_path, python_bin=CAD_PYTHON)
