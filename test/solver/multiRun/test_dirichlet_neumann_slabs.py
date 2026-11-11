# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The slabs of ``cases/coupled_slabs``, coupled Dirichlet-Neumann, reach the same profile.

``cases/dirichlet_neumann_slabs``: left (``coupledHeatFlux``) takes the heat flux
right takes in and sends its wall temperature; right (``coupledTemperature``)
takes it and sends its heat flux. Serial with Aitken relaxation around each
``solve_T``; tolerance as in ``test_coupled_slabs``.
"""

import numpy as np
import pytest
from framework.coupling.regions import couple

from neofoam.framework.context import Context
from neofoam.solver.multiRun import run
from solver.multiRun.case_files import copy_case
from solver.multiRun.slab_solver import CoupledHeatFlux, CoupledTemperature

EXPECTED = {
    "left": [390.625, 371.875, 353.125, 334.375],
    "right": [321.875, 315.625, 309.375, 303.125],
}


@pytest.fixture(scope="module")
def regions(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Context]:
    run_dir = copy_case("dirichlet_neumann_slabs", tmp_path_factory.mktemp("slabs") / "case")
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.chdir(run_dir)
        return run()


@pytest.mark.parametrize("region", ["left", "right"])
def test_dirichlet_neumann_slabs_reach_the_steady_profile(
    regions: dict[str, Context], region: str
) -> None:
    np.testing.assert_allclose(
        regions[region].fields["T"],
        EXPECTED[region],
        rtol=0,
        atol=1e-8,
        err_msg=f"dirichlet_neumann_slabs region {region}",
    )


def test_dirichlet_neumann_slabs_couple_a_heat_flux_with_a_temperature_side(
    regions: dict[str, Context],
) -> None:
    assert isinstance(regions["left"].models["patches"][1], CoupledHeatFlux)
    assert isinstance(regions["right"].models["patches"][0], CoupledTemperature)


def test_dirichlet_neumann_slabs_converge_in_a_few_aitken_passes(
    regions: dict[str, Context],
) -> None:
    controller = couple(regions["right"]).controller
    assert controller is not None

    assert [(o.passes, o.converged) for o in controller.outcomes] == [(5, True), (5, True)]
