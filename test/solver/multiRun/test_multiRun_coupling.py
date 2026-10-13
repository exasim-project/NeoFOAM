# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""multiRun couples two slab regions through one shared CouplingOp.

In ``cases/coupled_slabs`` the slab ``left`` (x in [0, 1], conductivity 1, 400 K
at x = 0) and the slab ``right`` (x in [1, 2], conductivity 3, 300 K at x = 2)
meet at x = 1, where left's patch ``right`` and right's patch ``left`` are
``coupledConduction``. The test injects the interface (``region_couplings``) in
place of the case's ``system/regionCoupling.yaml``: both patches share one
``CouplingOp`` with its defaults, which hand the data over unchanged (both
patches lie at x = 1) and stop once it changes by round-off only.

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
from neofoam.framework.coupling import CouplingOp
from neofoam.framework.graph import DAGResolver
from neofoam.framework.operations import IterativeOp, Operation, StepBuilder
from neofoam.framework.solver.registry import resolve_solver
from neofoam.solver.multiRun import multiRun, run
from solver.multiRun.slab_solver import slab

CASES = Path(__file__).parent / "cases"
# The package exports the multiRun solver under the module's name.
MULTI_RUN_MODULE = importlib.import_module("neofoam.solver.multiRun.multiRun")

Tree = list[Union[str, tuple[str, "Tree"]]]


def _with_slab(name: str) -> Any:
    return slab if name == "slab" else resolve_solver(name)


def _interface(regions: list[str]) -> dict[str, dict[str, CouplingOp]]:
    """Left's patch ``right`` and right's patch ``left`` share one coupling."""
    coupling = CouplingOp()
    return {"left": {"right": coupling}, "right": {"left": coupling}}


@pytest.fixture
def coupled_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shutil.copytree(CASES / "coupled_slabs", tmp_path / "coupled_slabs")
    monkeypatch.setattr(MULTI_RUN_MODULE, "resolve_solver", _with_slab)
    monkeypatch.setattr(MULTI_RUN_MODULE, "region_couplings", _interface)
    monkeypatch.chdir(tmp_path / "coupled_slabs")


@pytest.fixture(scope="module")
def ctx(tmp_path_factory: pytest.TempPathFactory) -> Context:
    run_dir = tmp_path_factory.mktemp("multiRun") / "coupled_slabs"
    shutil.copytree(CASES / "coupled_slabs", run_dir)
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(MULTI_RUN_MODULE, "resolve_solver", _with_slab)
        monkeypatch.setattr(MULTI_RUN_MODULE, "region_couplings", _interface)
        monkeypatch.chdir(run_dir)
        return run()


def _graph() -> StepBuilder:
    """The graph multiRun runs for the case in the working directory."""
    solver = multiRun.instantiate()
    ctx = solver.initialize()
    builder, model_ops = solver.execution_graph(ctx=ctx)
    return ctx.graph.apply(DAGResolver().resolve(builder, model_ops))


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
    ctx: Context, region: str, expected: list[float]
) -> None:
    np.testing.assert_allclose(
        ctx.models["regions"][region][1].fields["T"],
        expected,
        rtol=0,
        atol=1e-8,
        err_msg=f"coupled_slabs region {region}",
    )


@pytest.mark.usefixtures("coupled_case")
def test_multiRun_coupled_regions_share_one_loop() -> None:
    graph = _graph()

    assert _tree(graph.operations.ops) == [
        (
            "time_loop",
            [
                "set_time_step",
                "increment_time",
                "left.store_old_time",
                "right.store_old_time",
                (
                    "loop_solving_left.T_right.T",
                    [
                        "receive_left.T",
                        "left.solve_T",
                        "send_left.T",
                        "receive_right.T",
                        "right.solve_T",
                        "send_right.T",
                    ],
                ),
            ],
        )
    ]


@pytest.mark.usefixtures("coupled_case")
def test_multiRun_coupling_loop_leaves_the_old_time_outside() -> None:
    time_loop = _graph().operations.ops[0]

    *before_loop, coupling_loop = time_loop.sub_operations
    assert [op.operation_name for op in before_loop][-2:] == [
        "left.store_old_time",
        "right.store_old_time",
    ]
    assert all(
        "store_old_time" not in (op.operation_name or "") for op in coupling_loop.sub_operations
    )
