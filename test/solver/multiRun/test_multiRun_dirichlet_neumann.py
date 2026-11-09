# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The example of a coupled case: two slabs, coupled Dirichlet-Neumann.

``cases/dirichlet_neumann_slabs`` is ``cases/coupled_slabs`` with another pair
of coupled conditions at x = 1: slab ``left`` (conductivity 1, 400 K at x = 0)
is the Neumann side, ``coupledHeatFlux``: it takes the heat flux the right slab
takes in, and sends its wall temperature. Slab ``right`` (conductivity 3,
300 K at x = 2) is the Dirichlet side, ``coupledTemperature``: it takes that
wall temperature, and sends the heat flux through its wall. One model each way
(``WallTemperature``, ``WallHeatFlux``); Aitken relaxation of what right
receives as carried (serial: left steps first, so right gets left's wall
temperature of the same pass, left right's flux of the pass before). The serial
node wraps each region's ``solve_T``, so the time step stays outside it.

Each time step (1e6, long against the diffusion time) is a steady solve, so the
result is the steady profile of ``test_multiRun_coupling``: linear in each slab,
325 K at the interface.
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
from solver.multiRun.slab_solver import CoupledHeatFlux, CoupledTemperature, slab

CASE = Path(__file__).parent / "cases" / "dirichlet_neumann_slabs"
MULTI_RUN_MODULE = importlib.import_module("neofoam.solver.multiRun.multiRun")
EXPECTED = {
    "left": [390.625, 371.875, 353.125, 334.375],
    "right": [321.875, 315.625, 309.375, 303.125],
}


def _with_slab(name: str) -> Any:
    return slab if name == "slab" else resolve_solver(name)


@pytest.fixture(scope="module")
def regions(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Context]:
    run_dir = tmp_path_factory.mktemp("multiRun") / "case"
    shutil.copytree(CASE, run_dir)
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(MULTI_RUN_MODULE, "resolve_solver", _with_slab)
        monkeypatch.chdir(run_dir)
        return run()


@pytest.mark.parametrize("region", ["left", "right"])
def test_dirichlet_neumann_slabs_reach_the_steady_profile(
    regions: dict[str, Context], region: str
) -> None:
    np.testing.assert_allclose(regions[region].fields["T"], EXPECTED[region], rtol=0, atol=1e-8)


def test_dirichlet_neumann_slabs_exchange_one_model_each_way(regions: dict[str, Context]) -> None:
    left, right = (ctx.models["patches"][index] for ctx, index in zip(regions.values(), (1, 0)))
    assert isinstance(left, CoupledHeatFlux) and isinstance(right, CoupledTemperature)
    op = Coupling.of(regions["right"]).operations()[0].func
    assert isinstance(op, CouplingOp)
    controller = op.controller
    assert controller is not None
    # Aitken: a few passes per time step
    assert [(o.passes, o.converged) for o in controller.outcomes] == [(4, True), (4, True)]
