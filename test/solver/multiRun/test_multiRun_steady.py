# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""multiRun couples two steady regions: outer iterations, no time loop, no time step.

``cases/steady_pair`` runs the test-only solver ``steady`` (``steady_solver``) in
``a`` (``x = 1 + x_b``) and ``b`` (``x = 0.5 x_a``), coupled serial with the
identity mapping until ``x`` changes by < 1e-6 relative: the same fixed point
(a = 2, b = 1) and passes as ``affine_pair`` (``test_multiRun_affine``), with
three outer iterations instead of time steps. Nothing agrees on a time step and
the regions have no time: coupling needs no solution loop, and a serial node may
wrap the whole outer iteration (there is no time step it could hold).
"""

import importlib
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from neofoam.framework.context import Context
from neofoam.framework.coupling import Coupling, CouplingOp
from neofoam.framework.solver.registry import resolve_solver
from neofoam.solver.multiRun import run
from solver.multiRun.steady_solver import steady

CASE = Path(__file__).parent / "cases" / "steady_pair"
MULTI_RUN_MODULE = importlib.import_module("neofoam.solver.multiRun.multiRun")


@pytest.fixture(scope="module")
def regions(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Context]:
    run_dir = tmp_path_factory.mktemp("multiRun") / "case"
    shutil.copytree(CASE, run_dir)
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            MULTI_RUN_MODULE,
            "resolve_solver",
            lambda name: steady if name == "steady" else resolve_solver(name),
        )
        monkeypatch.chdir(run_dir)
        return run()


def _passes(regions: dict[str, Context], region: str) -> list[int]:
    """The solves of ``region`` in each outer iteration."""
    outer = [index for index, _x in regions[region].models["history"]]
    return [outer.count(index) for index in sorted(set(outer))]


def test_steady_regions_reach_the_fixed_point(regions: dict[str, Context]) -> None:
    np.testing.assert_allclose(regions["a"].fields["x"], [2.0], rtol=1e-6)
    np.testing.assert_allclose(regions["b"].fields["x"], [1.0], rtol=1e-6)


def test_steady_regions_couple_in_every_outer_iteration_without_a_time(
    regions: dict[str, Context],
) -> None:
    for region in ("a", "b"):
        assert regions[region].time is None
        assert "solution_loop" not in regions[region].models
        # the first outer iteration couples to the fixed point, the others start there;
        # each ends with its final pass
        assert _passes(regions, region) == [21, 2, 2]
    op: Any = Coupling.of(regions["b"]).operations()[0].func
    assert isinstance(op, CouplingOp) and op.controller is not None
    assert [(o.passes, o.converged) for o in op.controller.outcomes] == [
        (21, True),
        (2, True),
        (2, True),
    ]
