# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Two steady regions couple in every pass of the outer loop, without a time or a time step.

``cases/steady_pair`` is ``cases/affine_pair`` in the ``steady`` solver: the outer
loop's first pass takes the 21 iterations of ``test_affine_pair``, the others start
converged. The limit 1e-6 is relative: the fixed point holds to 1e-5.
"""

import numpy as np
import pytest
from framework.coupling.regions import Outcomes, couple

from neofoam.framework.context import Context
from neofoam.framework.execution_observer import observe
from neofoam.solver.multiRun import run
from solver.multiRun.case_files import copy_case


@pytest.fixture(scope="module")
def case(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, Context], Outcomes]:
    run_dir = copy_case("steady_pair", tmp_path_factory.mktemp("steady") / "case")
    outcomes = Outcomes()
    with pytest.MonkeyPatch.context() as monkeypatch, observe(outcomes):
        monkeypatch.chdir(run_dir)
        return run(), outcomes


@pytest.fixture(scope="module")
def regions(case: tuple[dict[str, Context], Outcomes]) -> dict[str, Context]:
    return case[0]


def _solves(regions: dict[str, Context], region: str) -> list[int]:
    """The solves of ``region`` in each pass of the outer loop."""
    outer = [index for index, _x in regions[region].models["history"]]
    return [outer.count(index) for index in sorted(set(outer))]


def test_steady_pair_reaches_the_fixed_point(regions: dict[str, Context]) -> None:
    final = [regions[region].fields["x"][0] for region in ("a", "b")]

    np.testing.assert_allclose(final, [2.0, 1.0], rtol=1e-5, err_msg="steady_pair")


@pytest.mark.parametrize("region", ["a", "b"])
def test_steady_pair_couples_in_every_pass_of_the_outer_loop_without_a_time(
    regions: dict[str, Context], region: str
) -> None:
    assert regions[region].time is None
    assert "solution_loop" not in regions[region].models
    assert _solves(regions, region) == [21, 2, 2]


def test_steady_pair_leader_records_an_outcome_per_pass_of_the_outer_loop(
    case: tuple[dict[str, Context], Outcomes],
) -> None:
    regions, outcomes = case
    leader = couple(regions["b"]).controller
    assert leader is not None

    assert [(o.iterations, o.converged) for o in outcomes.of(leader.label)] == [
        (21, True),
        (2, True),
        (2, True),
    ]
