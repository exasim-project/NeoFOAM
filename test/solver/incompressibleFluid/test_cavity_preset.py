# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The lid-driven-cavity preset is a complete, correctly-wired config set.

This is the contract the wizard / AI config-filling flows rely on: writing the
predefined configs + ``0/`` fields (no copied files) yields a case the solver can
actually run. We prove it by writing the preset, generating a *small* mesh
in-process with pybFoam blockMesh, and running a **single** solver iteration — a
missing or misconfigured config raises during init or the PIMPLE loop. Preset
parameters (nu, lid velocity, time control) flow through to the written configs.

Gated on pybFoam (in-process meshing + the NeoN solver); no native OpenFOAM
utilities are shelled out.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


import pybFoam as pyf  # noqa: E402

from neofoam.io import write_configs  # noqa: E402
from neofoam.solver.incompressibleFluid import run  # noqa: E402

from .case_presets import CAVITY_PATCHES, lid_driven_cavity  # noqa: E402

# A tiny 3x3x1 cavity — enough cells to wire every operation, cheap to step once.
# Patch names must match ``CAVITY_PATCHES``.
_BLOCKMESH = """\
FoamFile
{
    version 2.0;
    format ascii;
    class dictionary;
    object blockMeshDict;
}
convertToMeters 0.1;
vertices
(
    (0 0 0) (1 0 0) (1 1 0) (0 1 0)
    (0 0 0.1) (1 0 0.1) (1 1 0.1) (0 1 0.1)
);
blocks ( hex (0 1 2 3 4 5 6 7) (3 3 1) simpleGrading (1 1 1) );
edges ();
boundary
(
    movingWall { type wall; faces ((3 7 6 2)); }
    fixedWalls { type wall; faces ((0 4 7 3)(2 6 5 1)(1 5 4 0)); }
    frontAndBack { type empty; faces ((0 3 2 1)(4 5 6 7)); }
);
mergePatchPairs ();
"""


def _make_mesh(case: Path) -> None:
    (case / "system" / "blockMeshDict").write_text(_BLOCKMESH)
    runtime = pyf.Time(str(case.parent), case.name)
    block_dict = pyf.dictionary.read(str(case / "system" / "blockMeshDict"))
    pyf.meshing.generate_blockmesh(runtime, block_dict, False, "constant")


def test_cavity_preset_runs_one_iteration(tmp_path: Path) -> None:
    case = tmp_path / "cavity"
    case.mkdir()

    # Predefined configs + 0/ fields → disk. One step (end_time == delta_t).
    report = write_configs(
        lid_driven_cavity(end_time=0.001, delta_t=0.001), case_dir=case
    )
    assert set(report) == {
        "system/controlDict",
        "system/fvSchemes",
        "system/fvSolution",
        "constant/transportProperties",
        "constant/turbulenceProperties",
        "0/U",
        "0/p",
    }

    # fvSchemes must be on disk before blockMesh runs.
    _make_mesh(case)

    # The contract: the solver advances one step without a missing/mis-wired
    # config raising during init or the PIMPLE loop.
    prev = os.getcwd()
    os.chdir(case)
    try:
        run(["incompressibleFluid"])
    finally:
        os.chdir(prev)

    written = {
        p.name
        for p in case.iterdir()
        if p.is_dir() and p.name not in {"0", "constant", "system"}
    }
    assert written, "solver wrote no time directory — it did not advance"
    assert float(max(written, key=float)) == pytest.approx(0.001)


def test_preset_parameters_flow_into_configs() -> None:
    cfgs = {
        type(c).__name__: c
        for c in lid_driven_cavity(nu=0.25, lid_velocity=(2.0, 0.0, 0.0))
    }

    assert cfgs["TransportPropertiesConfig"].nu == 0.25
    u_dump = cfgs["UFieldConfig"].model_dump(by_alias=True)
    assert u_dump["boundaryField"]["movingWall"]["value"] == "uniform (2.0 0.0 0.0)"
    # Patches match what the mesh must provide.
    assert set(u_dump["boundaryField"]) == set(CAVITY_PATCHES)
