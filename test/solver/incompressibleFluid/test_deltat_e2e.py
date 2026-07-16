# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""End-to-end: the solver folds the model-owned timeStepConstraint to set deltaT."""

from pathlib import Path

import pytest

from neofoam.tooling.casebuild import from_template, block_mesh, patch
from neofoam.solver.incompressibleFluid import run

from .._run_case import cwd


def test_solver_constrains_delta_t_through_the_interface_fold(tmp_path: Path) -> None:
    repo_root = Path(__file__).parent.parent.parent.parent
    source = repo_root / "tutorials" / "pitzDaily"
    dt0 = 0.01
    end_time = dt0 * 3
    write_interval = dt0 * 3

    # Build base case from pitzDaily: copy, mesh, set timing and CFL control.
    base = (
        from_template(source)
        | block_mesh()
        | patch(
            "system/controlDict",
            {
                "endTime": end_time,
                "deltaT": dt0,
                "writeInterval": write_interval,
            },
        )
    ).build_at(tmp_path / "base")

    # Variant 1: CFL-constrained (adjustTimeStep on, maxCo 0.2).
    constrained = (
        base
        | patch(
            "system/controlDict",
            {"adjustTimeStep": True, "maxCo": 0.2},
        )
    ).build_at(tmp_path / "constrained")

    # Variant 2: Fixed timestep (adjustTimeStep off).
    fixed = (
        base
        | patch(
            "system/controlDict",
            {"adjustTimeStep": False},
        )
    ).build_at(tmp_path / "fixed")

    # Two real solver runs in ONE process (also the cross-run no-SIGBUS check).
    with cwd(constrained.path):
        ctx_constrained = run(["incompressibleFluid"])
    dt_constrained = float(ctx_constrained.time.delta_t)

    with cwd(fixed.path):
        ctx_fixed = run(["incompressibleFluid"])
    dt_fixed = float(ctx_fixed.time.delta_t)

    assert dt_fixed == pytest.approx(dt0)
    assert dt_constrained < dt0  # the interface fold shrank the step
