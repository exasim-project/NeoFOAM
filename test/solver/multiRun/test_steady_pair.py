# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Two steady regions couple in every outer iteration, without a time or a time step.

``cases/steady_pair`` is ``cases/affine_pair`` in the ``steady`` solver: three
outer iterations instead of time steps, coupled serial around the whole outer
iteration, which holds no time step. The first outer iteration takes the 21
passes of ``test_affine_pair``, the others start converged. The limit 1e-6 is
relative: the fixed point holds to 1e-5.
"""

import numpy as np
import pytest
from framework.coupling.regions import couple

from neofoam.framework.context import Context
from neofoam.solver.multiRun import run
from solver.multiRun.case_files import copy_case


@pytest.fixture(scope="module")
def regions(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Context]:
    run_dir = copy_case("steady_pair", tmp_path_factory.mktemp("steady") / "case")
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.chdir(run_dir)
        return run()


def _passes(regions: dict[str, Context], region: str) -> list[int]:
    """The solves of ``region`` in each outer iteration."""
    outer = [index for index, _x in regions[region].models["history"]]
    return [outer.count(index) for index in sorted(set(outer))]


def test_steady_pair_reaches_the_fixed_point(regions: dict[str, Context]) -> None:
    final = [regions[region].fields["x"][0] for region in ("a", "b")]

    np.testing.assert_allclose(final, [2.0, 1.0], rtol=1e-5, err_msg="steady_pair")


@pytest.mark.parametrize("region", ["a", "b"])
def test_steady_pair_couples_in_every_outer_iteration_without_a_time(
    regions: dict[str, Context], region: str
) -> None:
    assert regions[region].time is None
    assert "solution_loop" not in regions[region].models
    assert _passes(regions, region) == [21, 2, 2]


def test_steady_pair_leader_records_an_outcome_per_outer_iteration(
    regions: dict[str, Context],
) -> None:
    controller = couple(regions["b"]).controller
    assert controller is not None

    assert [(o.passes, o.converged) for o in controller.outcomes] == [
        (21, True),
        (2, True),
        (2, True),
    ]
