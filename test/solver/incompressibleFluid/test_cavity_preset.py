# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The lid-driven-cavity preset is a complete, correctly-wired config set.

This is the contract the wizard / AI config-filling flows rely on: writing the
predefined configs + ``0/`` fields (no copied files) yields a case the solver can
actually run. We prove it by building a case with the preset configs and a
committed blockMeshDict, meshing with blockMesh, and running a **single** solver
iteration — a missing or misconfigured config raises during init or the PIMPLE
loop. Preset parameters (nu, lid velocity, time control) flow through to the
written configs.

Gated on pybFoam (the NeoN solver); meshing via blockMesh (native OpenFOAM utility).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from neofoam.tooling.casebuild import from_template, block_mesh, configs
from neofoam.solver.incompressibleFluid import run

from .._run_case import cwd
from .case_presets import CAVITY_PATCHES, lid_driven_cavity


def test_cavity_preset_runs_one_iteration(tmp_path: Path) -> None:
    repo_root = Path(__file__).parent.parent.parent.parent
    cavity_template = (
        repo_root / "test" / "solver" / "incompressibleFluid" / "cases" / "cavity3x3"
    )

    # Build case: copy template (has blockMeshDict), write configs, then mesh.
    # Configs must be written before block_mesh() so fvSchemes/fvSolution are on disk.
    case = (
        from_template(cavity_template)
        | configs(*lid_driven_cavity(end_time=0.001, delta_t=0.001))
        | block_mesh()
    ).build_at(tmp_path / "cavity")

    # The preset is complete: exactly the expected config + field files land on disk
    # (configs() writes them; the on-disk set is the contract the wizard flows rely on).
    expected = {
        "system/controlDict",
        "system/fvSchemes",
        "system/fvSolution",
        "constant/transportProperties",
        "constant/turbulenceProperties",
        "0/U",
        "0/p",
    }
    assert expected <= {
        str(p.relative_to(case.path)) for p in case.path.rglob("*") if p.is_file()
    }

    # The contract: the solver advances one step without a missing/mis-wired
    # config raising during init or the PIMPLE loop.
    with cwd(case.path):
        run(["incompressibleFluid"])

    written = {
        p.name
        for p in case.path.iterdir()
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
