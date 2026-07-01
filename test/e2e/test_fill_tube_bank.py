# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The hardcoded config-driven driver authors a complete, runnable tube-bank case.

The fast check proves the redesign: filling the solver's ``BaseConfig`` models and
writing them lays down every required file (no hand-rendered text). The slow check
(gated on a sourced OpenFOAM) proves the authored case actually meshes + solves.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("pybFoam")

from neofoam.e2e.config import REQUIRED_FILES  # noqa: E402

# Load the driver, which lives under cases/ (not an importable package).
_DRIVER_PATH = Path(__file__).parent / "cases" / "fill_tube_bank.py"
_spec = importlib.util.spec_from_file_location("fill_tube_bank", _DRIVER_PATH)
assert _spec and _spec.loader
fill_tube_bank = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fill_tube_bank)


def _has_openfoam() -> bool:
    return bool(os.environ.get("WM_PROJECT_DIR"))


def test_authors_every_required_file(tmp_path: Path) -> None:
    """Filling + writing the configs lays down all 10 required case files."""
    case = fill_tube_bank.build_tube_bank(tmp_path)
    for rel in REQUIRED_FILES:
        assert (case / rel).is_file(), f"missing required file: {rel}"


def test_stages_geometry(tmp_path: Path) -> None:
    """The checked-in STLs are copied into constant/triSurface."""
    case = fill_tube_bank.build_tube_bank(tmp_path)
    staged = {p.name for p in (case / "constant" / "triSurface").glob("*.stl")}
    assert {
        "inlet.stl",
        "outlet.stl",
        "walls.stl",
        "tubes.stl",
        "frontBack.stl",
    } <= staged


def test_boundary_conditions_cover_every_patch(tmp_path: Path) -> None:
    """0/U has a boundary entry for every manifest patch (incl. snappy ``tubes``)."""
    case = fill_tube_bank.build_tube_bank(tmp_path)
    from neofoam.e2e.manifest import PatchManifest

    manifest = PatchManifest.load(fill_tube_bank.MANIFEST)
    u_text = (case / "0" / "U").read_text()
    for patch in manifest.patches:
        assert patch.name in u_text, f"0/U missing patch {patch.name}"
    # the inlet carries the fixed inlet velocity
    assert "fixedValue" in u_text and "uniform ( 1 0 0 )" in u_text


@pytest.mark.slow
@pytest.mark.skipif(
    not _has_openfoam(), reason="needs a sourced OpenFOAM (WM_PROJECT_DIR)"
)
def test_authored_case_meshes_and_solves(tmp_path: Path) -> None:
    """The authored case meshes (blockMesh→snappy→checkMesh) and solves to a time dir."""
    from neofoam.solver.incompressibleFluid.configs import ControlDictConfig

    case = fill_tube_bank.build_tube_bank(tmp_path)
    # Shorten the run so the smoke test is quick (write every step, stop after a few).
    ControlDictConfig(
        application="pimpleFoam",
        endTime=0.006,
        deltaT=0.002,
        writeControl="timeStep",
        writeInterval=1,
    ).save(case_dir=case)

    proc = subprocess.run(
        [sys.executable, "-m", "neofoam.e2e.solve", str(case)],
        capture_output=True,
        text=True,
    )
    time_dirs = sorted(
        child.name
        for child in case.iterdir()
        if child.is_dir()
        and child.name.replace(".", "", 1).isdigit()
        and float(child.name) > 0
    )
    assert time_dirs, (
        f"no time directory written\nSTDOUT:\n{proc.stdout[-1000:]}\n"
        f"STDERR:\n{proc.stderr[-1000:]}"
    )
