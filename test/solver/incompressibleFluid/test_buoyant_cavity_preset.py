# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""A laminar + Boussinesq case advances one step end to end.

Regression for the ``laminar`` + ``boussinesq`` run blocker: the energy equation
(``boussinesq.solve_energy``) reads the turbulence model's ``nut()``. On
``incompressibleFluid`` every model is built through the pybFoam fallback handle,
whose ``has_nut()`` is always ``True`` (the OpenFOAM laminar model exposes an eddy
viscosity that is identically zero), so ``solve_energy`` computes
``alphat = nut/Prt = 0`` and ``alpha_eff = nu/Pr`` — the correct molecular thermal
diffusivity. This test builds the smallest complete laminar-Boussinesq case
(:func:`buoyant_cavity`) on the committed cavity mesh and runs a single step; the
original blocker crashed on step 1 during the time loop.

Gated on pybFoam (the NeoN solver); meshing via blockMesh (native OpenFOAM utility).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("pybFoam")

from neofoam.tooling.casebuild import from_template, block_mesh, configs  # noqa: E402
from neofoam.solver.incompressibleFluid import run  # noqa: E402

from .._run_case import cwd  # noqa: E402
from .case_presets import buoyant_cavity  # noqa: E402

# Boussinesq builds intermediate fields whose first-iteration deltas legitimately
# underflow; disable OpenFOAM FP-exception trapping for the in-process run (matches
# ``test_hotRoom_comparison``).
os.environ["FOAM_SIGFPE"] = ""


def test_laminar_boussinesq_advances_one_step(tmp_path: Path) -> None:
    repo_root = Path(__file__).parent.parent.parent.parent
    cavity_template = (
        repo_root / "test" / "solver" / "incompressibleFluid" / "cases" / "cavity3x3"
    )

    case = (
        from_template(cavity_template)
        | configs(*buoyant_cavity(end_time=0.001, delta_t=0.001))
        | block_mesh()
    ).build_at(tmp_path / "buoyant_cavity")

    # The buoyancy configs land on disk alongside pimple's (the case is complete).
    expected = {
        "constant/g",
        "0/T",
        "0/p_rgh",
        "0/alphat",
    }
    on_disk = {
        str(p.relative_to(case.path)) for p in case.path.rglob("*") if p.is_file()
    }
    assert expected <= on_disk, f"missing: {expected - on_disk}"

    # The contract: native laminar + boussinesq advances one step. Before the fix
    # this raised AttributeError in solve_energy during the time loop.
    log = case.path / "solver.log"
    with cwd(case.path):
        run(["incompressibleFluid"], log_file=log)

    output = log.read_text() if log.exists() else ""

    # The energy equation actually solved (the code path that used to crash).
    assert "Solving for T" in output, f"energy equation did not run.\nLog:\n{output}"

    written = {
        p.name
        for p in case.path.iterdir()
        if p.is_dir() and p.name not in {"0", "constant", "system"}
    }
    assert written, (
        f"solver wrote no time directory — it did not advance.\nLog:\n{output}"
    )
    assert float(max(written, key=float)) == pytest.approx(0.001)
