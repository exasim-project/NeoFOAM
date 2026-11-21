# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Two slabs coupled at x = 1 reach the steady profile of conduction in series.

``cases/coupled_slabs``: slab ``left`` (x in [0, 1], conductivity 1, 400 K at
x = 0) and slab ``right`` (x in [1, 2], conductivity 3, 300 K at x = 2), their
patches at x = 1 ``coupledConduction``, coupled parallel around the whole time
step. The time step (1e6) is long against the diffusion time (1), so each step
is a steady solve: linear in each slab, 325 K at the interface, which finite
volumes reproduce exactly at the cell centres. The coupling stops at a relative
change of 1e-12, so 1e-8 K bounds what is left.
"""

import numpy as np
import pytest
from framework.coupling.regions import tree

from neofoam.framework.context import Context
from neofoam.framework.execution_observer import ExecutionObserver, observe
from neofoam.framework.operations import Operations
from neofoam.solver.multiRun import run
from solver.multiRun.case_files import copy_case


class _Graphs(ExecutionObserver):
    """The graph each region resolved, by region."""

    def __init__(self) -> None:
        self.trees: dict[object, list[object]] = {}

    def operations(self, operations: Operations, ctx: Context) -> None:
        self.trees[ctx.region] = tree(operations.ops)


@pytest.fixture(scope="module")
def observed(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, Context], _Graphs]:
    graphs = _Graphs()
    with pytest.MonkeyPatch.context() as monkeypatch, observe(graphs):
        monkeypatch.chdir(copy_case("coupled_slabs", tmp_path_factory.mktemp("slabs") / "case"))
        return run(), graphs


@pytest.mark.parametrize(
    ("region", "expected"),
    [
        ("left", [390.625, 371.875, 353.125, 334.375]),
        ("right", [321.875, 315.625, 309.375, 303.125]),
    ],
)
def test_coupled_slabs_reach_the_steady_profile(
    observed: tuple[dict[str, Context], _Graphs], region: str, expected: list[float]
) -> None:
    regions, _graphs = observed

    np.testing.assert_allclose(
        regions[region].fields["T"],
        expected,
        rtol=0,
        atol=1e-8,
        err_msg=f"coupled_slabs region {region}",
    )


@pytest.mark.parametrize("region", ["left", "right"])
def test_coupled_slabs_wrap_the_whole_time_step_in_the_coupling_op(
    observed: tuple[dict[str, Context], _Graphs], region: str
) -> None:
    _regions, graphs = observed

    assert graphs.trees[region] == [
        (
            "time_loop",
            [
                (
                    f"{region}.coupling",
                    [
                        "set_time_step",
                        f"{region}.time_step",
                        "increment_time",
                        "store_old_time",
                        "solve_T",
                    ],
                )
            ],
        )
    ]
