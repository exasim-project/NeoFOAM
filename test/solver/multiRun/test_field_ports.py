# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""A field port of ``system/coupling.yaml`` is a sensor: no Python, the yaml names it.

``cases/field_ports`` is ``cases/coupled_slabs`` with a third region, the probe
``meter``. The field port ``T_cells`` reads slab ``left``'s temperature per cell,
``volume-mean`` reduces it onto meter's Global port ``T_left``, and meter solves
``x = T_left``. The slabs reach the steady profile of ``test_coupled_slabs``,
whose left cells 390.625, 371.875, 353.125 and 334.375 K (equal volumes) have
the mean 362.5 K; the coupling stops at a relative change of 1e-12, so 1e-8 K
bounds what is left.
"""

from pathlib import Path

import numpy as np
import pytest

from neofoam.framework.context import Context
from neofoam.solver.multiRun import run
from solver.multiRun.case_files import copy_case


@pytest.fixture(scope="module")
def regions(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Context]:
    with pytest.MonkeyPatch.context() as monkeypatch:
        case: Path = copy_case("field_ports", tmp_path_factory.mktemp("ports") / "case")
        monkeypatch.chdir(case)
        return run()


def test_a_field_port_on_the_cells_senses_the_volume_mean_of_its_field(
    regions: dict[str, Context],
) -> None:
    (_index, x), *_rest = reversed(regions["meter"].models["history"])

    np.testing.assert_allclose(x, 362.5, rtol=0, atol=1e-8, err_msg="field_ports meter x")


def test_a_field_port_only_reads_the_field_it_senses(regions: dict[str, Context]) -> None:
    np.testing.assert_allclose(
        regions["left"].fields["T"],
        [390.625, 371.875, 353.125, 334.375],
        rtol=0,
        atol=1e-8,
        err_msg="field_ports region left",
    )
