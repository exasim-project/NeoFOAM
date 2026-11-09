# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""multiRun couples two regions that only do arithmetic, to test the framework's data exchange.

``cases/affine_pair`` runs the test-only solver ``affine`` (``affine_solver``) in
the regions ``a`` and ``b``, each with its own ``constant/<region>/affine.yaml``:

    a = 1.0 * b_from_b + 1.0        b = 0.5 * a_from_a + 0.0

``system/regionCoupling.yaml`` couples their patches ``interface`` (serial,
relaxation 1, relative change of ``x`` < 1e-6) in one node that wraps each
region's ``solve_x`` (a serial node leaves the time step outside): each region
runs on its own thread and repeats its ``solve_x`` in its coupling op
``<region>.coupling`` to the fixed point ``a = 2``, ``b = 1``, as b (the leader)
decides. Both start at 0.
Serial, b receives what a sent in the same pass (Gauss-Seidel), so pass k gives
``a_k = 2 - 2**(1 - k)`` and ``b_k = 1 - 2**-k``: the error halves every pass.
Before pass k + 1 the leader checks the relative change of what both sides sent in pass k,
``2**(1 - k) / a_k`` for a, which drops below 1e-6 at k = 20; pass 21 is the
final one, after which it stops. The case runs two time steps; the second starts
converged and stops after its final pass, the second.
"""

import importlib
import shutil
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pytest

from neofoam.framework.context import Context
from neofoam.framework.coupling import (
    AitkenRelaxation,
    ConstantRelaxation,
    Coupling,
    CouplingAcceleration,
    CouplingNotConvergedWarning,
)
from neofoam.framework.coupling.controller import CouplingController, CouplingOutcome
from neofoam.framework.coupling.loop import CouplingOp
from neofoam.framework.execution_observer import ExecutionObserver, observe
from neofoam.framework.operations import Operation, Operations
from neofoam.framework.solver.registry import resolve_solver
from neofoam.solver.multiRun import run
from neofoam.solver.multiRun.configs import RegionCouplingConfig
from solver.multiRun.affine_solver import affine

CASE = Path(__file__).parent / "cases" / "affine_pair"
# The package exports the multiRun solver under the module's name.
MULTI_RUN_MODULE = importlib.import_module("neofoam.solver.multiRun.multiRun")
# 20 passes to converge, then the final pass
PASSES = 21


def _with_affine(name: str) -> Any:
    return affine if name == "affine" else resolve_solver(name)


def _case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    acceleration: Optional[CouplingAcceleration] = None,
    **update: Any,
) -> None:
    """The case in ``tmp_path`` as the working directory, its coupling node updated.

    ``update`` sets entries of the node (``scheme``, ``maxIterations``),
    ``acceleration`` that of its exchange.
    """
    shutil.copytree(CASE, tmp_path / "case")
    if update or acceleration:
        config = RegionCouplingConfig.load(case_dir=tmp_path / "case")
        assert config.couplings is not None
        (node,) = config.couplings
        if acceleration is not None:
            (exchange,) = node.exchanges
            update["exchanges"] = [exchange.model_copy(update={"acceleration": acceleration})]
        node = node.model_copy(update=update)
        config.model_copy(update={"couplings": [node]}).save(case_dir=tmp_path / "case")
    monkeypatch.setattr(MULTI_RUN_MODULE, "resolve_solver", _with_affine)
    monkeypatch.chdir(tmp_path / "case")


Regions = dict[str, Context]


def _history(regions: Regions, region: str, time_index: int) -> list[float]:
    return [x for index, x in regions[region].models["history"] if index == time_index]


def _coupling(regions: Regions, region: str) -> CouplingOp:
    op = Coupling.of(regions[region]).operations()[0].func
    assert isinstance(op, CouplingOp)
    return op


@pytest.fixture(scope="module")
def ctx(tmp_path_factory: pytest.TempPathFactory) -> Regions:
    with pytest.MonkeyPatch.context() as monkeypatch:
        _case(tmp_path_factory.mktemp("affine"), monkeypatch)
        return run()


def test_multiRun_affine_reads_each_regions_config(ctx: Regions) -> None:
    assert list(ctx) == ["a", "b"]
    for region, gain, offset in [("a", 1.0, 1.0), ("b", 0.5, 0.0)]:
        region_ctx = ctx[region]
        assert region_ctx.region == region
        assert (region_ctx.models["affine"].gain, region_ctx.models["affine"].offset) == (
            gain,
            offset,
        )
        assert region_ctx.time.index == 2


def test_multiRun_affine_regions_couple_through_their_coupling_ops(ctx: Regions) -> None:
    a_op, b_op = _coupling(ctx, "a"), _coupling(ctx, "b")
    assert [(op.label, op.region, op.is_leader) for op in (a_op, b_op)] == [
        ("a.coupling", "a", False),
        ("b.coupling", "b", True),
    ]
    assert a_op.controller is None and b_op.controller is not None
    (a_side,), (b_side,) = a_op.sides, b_op.sides
    assert [(side.end, side.field) for side in (a_side, b_side)] == [
        (("a", "interface"), "x"),
        (("b", "interface"), "x"),
    ]
    assert (a_op.partner(a_side), b_op.partner(b_side)) == ("b.interface", "a.interface")
    exchange = a_op.exchange_of(a_side)
    assert (exchange.name, exchange.mapping.type) == ("a-b", "nearest-neighbor")
    assert b_op.exchange_of(b_side).name == "a-b"


def test_multiRun_affine_exchanges_the_partners_value_every_pass(ctx: Regions) -> None:
    a = np.array(_history(ctx, "a", 1))
    b = np.array(_history(ctx, "b", 1))
    k = np.arange(1, PASSES + 1)

    np.testing.assert_allclose(a, 2 - 2.0 ** (1 - k), rtol=0, atol=1e-15)
    np.testing.assert_allclose(b, 1 - 2.0**-k, rtol=0, atol=1e-15)
    # what each side received is what the other sent: a from b's last pass, b from a's same pass
    np.testing.assert_allclose(a[1:], b[:-1] + 1.0, rtol=0, atol=1e-15)
    np.testing.assert_allclose(b, 0.5 * a, rtol=0, atol=1e-15)


def test_multiRun_affine_converges_to_the_fixed_point(ctx: Regions) -> None:
    np.testing.assert_allclose(ctx["a"].fields["x"], [2.0], rtol=1e-6)
    np.testing.assert_allclose(ctx["b"].fields["x"], [1.0], rtol=1e-6)


def test_multiRun_affine_iteration_count(ctx: Regions) -> None:
    for region in ["a", "b"]:
        assert len(_history(ctx, region, 1)) == PASSES
        # the second time step starts at the fixed point: one pass shows no change,
        # then the final pass
        assert len(_history(ctx, region, 2)) == 2


@pytest.mark.parametrize(
    ("scheme", "acceleration", "passes"),
    [
        ("serial", AitkenRelaxation(relaxation=0.5), 5),
        # Jacobi: each side receives what the other sent in the previous pass
        ("parallel", ConstantRelaxation(relaxation=1.0), 40),
    ],
    ids=["serial-aitken", "parallel-constant"],
)
def test_multiRun_affine_scheme_and_acceleration_change_only_the_passes(
    scheme: str,
    acceleration: CouplingAcceleration,
    passes: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _case(tmp_path, monkeypatch, scheme=scheme, acceleration=acceleration)

    ctx = run()

    np.testing.assert_allclose(ctx["a"].fields["x"], [2.0], rtol=1e-5)
    np.testing.assert_allclose(ctx["b"].fields["x"], [1.0], rtol=1e-5)
    assert len(_history(ctx, "a", 1)) == passes


def test_multiRun_affine_records_the_coupling_outcome_of_every_time_step(ctx: Regions) -> None:
    controller = _coupling(ctx, "b").controller
    assert controller is not None
    outcomes = controller.outcomes

    assert [(o.passes, o.converged) for o in outcomes] == [(PASSES, True), (2, True)]
    # the final pass 21 changed a by 2**-20 (of 2) and b by 2**-21 (of 1)
    assert outcomes[0].residuals == {"x": pytest.approx(2.0**-21, rel=1e-6)}
    assert outcomes[1].residuals["x"] < 1e-6


def test_multiRun_affine_stops_at_max_iterations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _case(tmp_path, monkeypatch, maxIterations=5)

    with pytest.warns(CouplingNotConvergedWarning, match="did not converge in 5 passes"):
        ctx = run()

    # unconverged after 5 passes, the time step goes on with a_5
    assert len(_history(ctx, "a", 1)) == 5
    assert _history(ctx, "a", 1)[-1] == 2 - 2.0**-4


def test_multiRun_affine_a_serial_node_around_the_time_step_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without ``wraps`` the serial node would wrap the whole time step: it raises, not hangs."""
    _case(tmp_path, monkeypatch, wraps={})

    with pytest.raises(
        ValueError,
        match=r"coupling 'coupling' is serial but holds the time step of its regions; make it "
        r"parallel or place it around named operations \(wraps\.[ab]\.operations\)",
    ):
        run()


class _Recorder(ExecutionObserver):
    """What every region reported, in the order of its own thread."""

    def __init__(self) -> None:
        self.graphs: dict[Any, Operations] = {}
        self.init_orders: list[list[tuple[str, Any]]] = []
        self.ran: list[tuple[str, Context]] = []
        self.tests: dict[Any, list[tuple[str, int, bool]]] = {"a": [], "b": []}
        self.couplings: list[tuple[Any, CouplingOutcome]] = []

    def coupling_finished(self, coupling: CouplingController, outcome: CouplingOutcome) -> None:
        self.couplings.append((coupling.label, outcome))

    def init_steps(self, steps: list[Any]) -> None:
        self.init_orders.append([(step.name, step.region) for step in steps])

    def operations(self, operations: Operations, ctx: Context) -> None:
        self.graphs[ctx.region] = operations

    def before_operation(self, op: Operation, ctx: Context) -> None:
        self.ran.append((op.name or "", ctx))

    def after_loop_test(self, op: Operation, ctx: Context, iteration: int, result: bool) -> None:
        self.tests[ctx.region].append((op.name or "", iteration, result))


def _tree(ops: list[Operation]) -> list[Any]:
    return [(op.name, _tree(op.sub_operations)) if op.sub_operations else op.name for op in ops]


def test_multiRun_affine_execution_observer_sees_each_region_s_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _case(tmp_path, monkeypatch)
    recorder = _Recorder()

    with observe(recorder):
        regions = run()

    # every region initializes on its own thread, its steps tagged with the region
    assert sorted(
        sorted({region for _name, region in order}) for order in recorder.init_orders
    ) == [
        ["a"],
        ["b"],
    ]
    # a graph per region: its own time loop, its time-step op, its solve in its coupling op
    assert {region: _tree(graph.ops) for region, graph in recorder.graphs.items()} == {
        region: [
            (
                "time_loop",
                [
                    "set_time_step",
                    f"{region}.time_step",
                    "increment_time",
                    (f"{region}.coupling", ["solve_x"]),
                ],
            )
        ]
        for region in ("a", "b")
    }
    # a region's operations run on the region's context
    assert {(name, ctx.region) for name, ctx in recorder.ran} == {
        (name, region)
        for region in ("a", "b")
        for name in (
            "set_time_step",
            f"{region}.time_step",
            "increment_time",
            "solve_x",
        )
    }
    assert all(ctx is regions[ctx.region or ""] for _name, ctx in recorder.ran)
    # every region's coupling op runs PASSES passes, then the leader's decision ends
    # it in the same pass; the second time step stops after two passes
    for region in ("a", "b"):
        couple = [t for t in recorder.tests[region] if t[0] == f"{region}.coupling"]
        assert couple == [
            *[(f"{region}.coupling", k, True) for k in range(1, PASSES + 1)],
            (f"{region}.coupling", PASSES + 1, False),
            (f"{region}.coupling", 1, True),
            (f"{region}.coupling", 2, True),
            (f"{region}.coupling", 3, False),
        ]
    # b's coupling op, the leader, decided
    assert [(label, o.passes, o.converged) for label, o in recorder.couplings] == [
        ("b.coupling", PASSES, True),
        ("b.coupling", 2, True),
    ]


def test_multiRun_affine_regions_step_the_smallest_time_step_any_asks_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """a asks for deltaT 0.5, b for the controlDict's 1.0: both step 0.5 (the time-step op)."""
    _case(tmp_path, monkeypatch)
    affine_a = Path("constant/a/affine.yaml")
    affine_a.write_text(affine_a.read_text() + "deltaT: 0.5\n")

    regions = run()

    for region in ("a", "b"):
        time = regions[region].time
        assert (time.value, time.index, time.delta_t) == (2.0, 4, 0.5)
        assert sorted({index for index, _x in regions[region].models["history"]}) == [1, 2, 3, 4]
