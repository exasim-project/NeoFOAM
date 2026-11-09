# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""multiRun couples two slab regions through one exchange.

In ``cases/coupled_slabs`` the slab ``left`` (x in [0, 1], conductivity 1, 400 K
at x = 0) and the slab ``right`` (x in [1, 2], conductivity 3, 300 K at x = 2)
meet at x = 1, where left's patch ``right`` and right's patch ``left`` are
``coupledConduction``. The test couples the patches (multiRun's ``couplings``) in
place of the case's ``system/regionCoupling.yaml``: one exchange with the
``identity`` mapping (both patches lie at x = 1), the default scheme (parallel)
and the default ``CouplingController``, which stops once the data changes by
round-off only.

The time step (1e6) is long against the diffusion time (1), so each
backward-Euler step is a steady solve: the first leaves ~6e-6 K of the initial
field, the second ~1e-11 K. Steady conduction through two slabs in series is
linear in each slab, with the interface at (1*400 + 3*300) / (1 + 3) = 325 K, and
a finite-volume solution of a linear profile is exact: the expected cell values
are that profile at the cell centres. The coupling converges by ~0.5 per pass
and stops at a relative change of 1e-12, so 1e-8 K bounds what is left.
One multiRun run serves the field tests, hence the module-scoped fixture.
"""

import importlib
import shutil
from pathlib import Path
from typing import Any, Union

import numpy as np
import pytest

from neofoam.framework.context import Context
from neofoam.framework.coupling import (
    CommChannel,
    CouplingNode,
    Link,
    RegionCoupling,
    region_couplings,
)
from neofoam.framework.execution_observer import ExecutionObserver, observe
from neofoam.framework.operations import IterativeOp, Operation, Operations
from neofoam.framework.solver.registry import resolve_solver
from neofoam.solver.multiRun import run
from solver.multiRun.slab_solver import slab

CASES = Path(__file__).parent / "cases"
# The package exports the multiRun solver under the module's name.
MULTI_RUN_MODULE = importlib.import_module("neofoam.solver.multiRun.multiRun")

Tree = list[Union[str, tuple[str, "Tree"]]]


def _with_slab(name: str) -> Any:
    return slab if name == "slab" else resolve_solver(name)


def _interface(regions: list[str], comm_channel: CommChannel) -> dict[str, RegionCoupling]:
    """Left's patch ``right`` and right's patch ``left`` in one exchange."""
    link = Link("interface", (("left", "right"), ("right", "left")))
    return region_couplings(
        regions, [link], comm_channel, nodes=[CouplingNode("coupling", ("interface",))]
    )


def _case(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shutil.copytree(CASES / "coupled_slabs", run_dir)
    monkeypatch.setattr(MULTI_RUN_MODULE, "resolve_solver", _with_slab)
    monkeypatch.setattr(MULTI_RUN_MODULE, "couplings", _interface)
    monkeypatch.chdir(run_dir)


class _Graphs(ExecutionObserver):
    """The graph each region resolved, by region."""

    def __init__(self) -> None:
        self.trees: dict[Any, Tree] = {}

    def operations(self, operations: Operations, ctx: Context) -> None:
        self.trees[ctx.region] = _tree(operations.ops)


@pytest.fixture(scope="module")
def observed(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, Context], _Graphs]:
    graphs = _Graphs()
    with pytest.MonkeyPatch.context() as monkeypatch, observe(graphs):
        _case(tmp_path_factory.mktemp("multiRun") / "coupled_slabs", monkeypatch)
        return run(), graphs


def _tree(ops: list[Operation]) -> Tree:
    return [
        (op.operation_name or "", _tree(op.sub_operations))
        if isinstance(op.func, IterativeOp)
        else op.operation_name or ""
        for op in ops
    ]


@pytest.mark.parametrize(
    ("region", "expected"),
    [
        ("left", [390.625, 371.875, 353.125, 334.375]),
        ("right", [321.875, 315.625, 309.375, 303.125]),
    ],
)
def test_multiRun_coupled_regions_reach_the_steady_profile(
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
def test_multiRun_each_region_wraps_its_solution_in_its_coupling_op(
    observed: tuple[dict[str, Context], _Graphs], region: str
) -> None:
    """The coupling op (parallel, at the top by default) wraps the whole time step; the
    regions agree on deltaT before the time advances (the time-step op) and the old time
    is stored, each once per time step (``first``)."""
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
