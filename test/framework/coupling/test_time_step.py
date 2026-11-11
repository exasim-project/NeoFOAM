# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The time-step op: coupled transient regions step the smallest time step any asks for.

Each region's graph is a time loop around ``set_time_step`` (the region asks for
its time step, ``ctx.time.delta_t``) and ``increment_time`` (logs the time step
it advances by), both ``first`` as the solutionLoop's, with the region's
time-step op (``Coupling.time_step``) merged in. The regions run on their own
threads; the last region leads the group.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Literal, Optional

import pytest
from framework.coupling.regions import contexts
from framework.coupling.regions import transport as transport_of

from neofoam.framework.context import Context
from neofoam.framework.coupling import Coupling, Link
from neofoam.framework.coupling.region import run_regions
from neofoam.framework.graph import DAGResolver
from neofoam.framework.operations import (
    IterativeOp,
    Operation,
    Operations,
    SequentialOp,
    StepBuilder,
)
from neofoam.framework.types import OperationMetadata, Placement

CHAIN = [Link("a-b", (("a", "p"), ("b", "p"))), Link("b-c", (("b", "q"), ("c", "p")))]


@dataclass
class _Time:
    """Stands in for the solutionLoop's ``ctx.time``."""

    delta_t: float


class _Passes:
    """Loop predicate: ``count`` passes per run of the loop."""

    def __init__(self, count: int) -> None:
        self.count = count
        self.done = 0

    def __call__(self, ctx: Context) -> bool:
        self.done += 1
        if self.done <= self.count:
            return True
        self.done = 0
        return False


def _op(name: str, func: object, first: bool = False) -> Operation:
    return Operation(
        func=SequentialOp(func),  # type: ignore[arg-type]
        metadata=OperationMetadata(op_name=name, first=first),
    )


def _time_loop(
    asked: list[float],
    log: list[float],
    solves: Optional[list[int]] = None,
    inner: bool = False,
) -> StepBuilder:
    """A time loop of ``len(asked)`` steps: set_time_step asks for ``asked[step]``.

    With ``inner`` the solve runs in an ``inner_loop`` stepped after
    increment_time, as incompressibleFluid's PIMPLE loop.
    """
    steps = iter(asked)

    def set_time_step(ctx: Context) -> None:
        ctx.time.delta_t = next(steps)

    builder = StepBuilder()
    loop = Operation(
        func=IterativeOp(_Passes(len(asked))), metadata=OperationMetadata(op_name="time_loop")
    )
    with builder.loop(loop) as time_builder:
        time_builder.step(_op("set_time_step", set_time_step, first=True))
        time_builder.step(
            _op("increment_time", lambda ctx: log.append(ctx.time.delta_t), first=True)
        )
        if solves is not None and inner:
            inner_loop = Operation(
                func=IterativeOp(_Passes(2)), metadata=OperationMetadata(op_name="inner_loop")
            )
            with time_builder.loop(inner_loop) as inner_builder:
                inner_builder.step(_op("solve", lambda ctx: solves.append(1)))
        elif solves is not None:
            time_builder.step(_op("solve", lambda ctx: solves.append(1)))
    return builder


def _graph(ctx: Context, builder: StepBuilder, *loops: Operation) -> Operations:
    time_step = Coupling.of(ctx).time_step(ctx, builder.operations)
    assert time_step is not None
    return DAGResolver().resolve(builder, Operations([time_step, *loops])).operations


def _step(
    asked: dict[str, list[float]], passes: int = 0
) -> tuple[dict[str, Context], dict[str, list[float]], dict[str, list[int]]]:
    """Run every region of the chain a - b - c over the steps ``asked``; a coupling-like
    loop of ``passes`` passes around each time step, if any."""
    regions = contexts(list(asked), CHAIN)
    logs: dict[str, list[float]] = {region: [] for region in asked}
    solves: dict[str, list[int]] = {region: [] for region in asked}
    graphs = {}
    for region, ctx in regions.items():
        ctx.time = _Time(1.0)
        loops = []
        if passes:
            loops.append(
                Operation(
                    func=IterativeOp(_Passes(passes)),
                    metadata=OperationMetadata(op_name=f"{region}.couple", wraps=Placement()),
                )
            )
        graphs[region] = _graph(
            ctx, _time_loop(asked[region], logs[region], solves[region]), *loops
        )
    run_regions(
        {region: partial(graphs[region].run, regions[region]) for region in asked},
        transport_of(regions["a"]),
    )
    return regions, logs, solves


def test_the_regions_step_the_smallest_time_step_any_asks_for() -> None:
    asked = {"a": [1.0, 0.2], "b": [0.5, 0.3], "c": [2.0, 0.4]}

    regions, logs, _solves = _step(asked)

    assert logs == {region: [0.5, 0.2] for region in asked}
    assert {region: ctx.time.delta_t for region, ctx in regions.items()} == {
        region: 0.2 for region in asked
    }


def test_the_time_step_op_runs_between_set_time_step_and_increment_time() -> None:
    regions = contexts(["a", "b", "c"], CHAIN)
    ctx = regions["a"]
    ctx.time = _Time(1.0)

    (loop,) = _graph(ctx, _time_loop([1.0], [], [])).ops

    assert [op.operation_name for op in loop.sub_operations] == [
        "set_time_step",
        "a.time_step",
        "increment_time",
        "solve",
    ]
    assert loop.sub_operations[1].metadata.first


def test_the_time_step_op_leaves_a_loop_stepped_after_increment_time_after_it() -> None:
    """The time-step op changes the order of a PIMPLE-like step only by its own insertion."""
    regions = contexts(["a", "b", "c"], CHAIN)
    ctx = regions["a"]
    ctx.time = _Time(1.0)

    (loop,) = _graph(ctx, _time_loop([1.0], [], [], inner=True)).ops

    assert [op.operation_name for op in loop.sub_operations] == [
        "set_time_step",
        "a.time_step",
        "increment_time",
        "inner_loop",
    ]


def test_inside_a_coupling_loop_the_time_step_is_agreed_once_per_time_step() -> None:
    asked = {"a": [1.0, 0.2], "b": [0.5, 0.3], "c": [2.0, 0.4]}

    _regions, logs, solves = _step(asked, passes=3)

    assert logs == {region: [0.5, 0.2] for region in asked}
    assert solves == {region: [1] * 6 for region in asked}


def test_a_steady_region_has_no_time_step_op() -> None:
    ctx = contexts(["a", "b", "c"], CHAIN)["a"]

    assert ctx.time is None
    assert Coupling.of(ctx).time_step(ctx, _time_loop([1.0], []).operations) is None


def test_a_region_the_case_does_not_couple_has_no_time_step_op() -> None:
    ctx = contexts(["a", "b", "c"], CHAIN[:1])["c"]
    ctx.time = _Time(1.0)

    assert Coupling.of(ctx).time_step(ctx, _time_loop([1.0], []).operations) is None


def test_a_transient_coupled_region_without_the_solution_loop_ops_raises() -> None:
    ctx = contexts(["a", "b", "c"], CHAIN)["a"]
    ctx.time = _Time(1.0)
    builder = StepBuilder()
    builder.step(_op("set_time_step", lambda ctx: None))

    with pytest.raises(
        ValueError,
        match=r"region a steps in time coupled with \['a', 'b', 'c'\], but its graph has no "
        r"\['increment_time'\]",
    ):
        Coupling.of(ctx).time_step(ctx, builder.operations)


def _coupling_loop(region: str, placement: Placement) -> Operation:
    """A loop standing in for the coupling op of the region's node ``coupling``."""
    return Operation(
        func=IterativeOp(_Passes(1)),
        metadata=OperationMetadata(op_name=f"{region}.coupling", wraps=placement),
    )


def test_a_serial_node_holding_the_time_step_raises() -> None:
    ctx = contexts(["a", "b", "c"], CHAIN, scheme="serial")["a"]
    ctx.time = _Time(1.0)
    ops = _graph(ctx, _time_loop([1.0], [], []), _coupling_loop("a", Placement()))

    with pytest.raises(
        ValueError,
        match=r"coupling 'coupling' is serial but holds the time step of its regions; make it "
        r"parallel or place it around named operations \(wraps\.a\.operations\)",
    ):
        Coupling.of(ctx).check_graph(ops)


@pytest.mark.parametrize(
    ("scheme", "placement"),
    [("parallel", Placement()), ("serial", Placement(("solve",)))],
    ids=["parallel at the top", "serial around solve"],
)
def test_a_parallel_node_or_one_around_named_operations_may_hold_the_time_step(
    scheme: Literal["serial", "parallel"], placement: Placement
) -> None:
    ctx = contexts(["a", "b", "c"], CHAIN, scheme=scheme)["a"]
    ctx.time = _Time(1.0)
    ops = _graph(ctx, _time_loop([1.0], [], []), _coupling_loop("a", placement))

    Coupling.of(ctx).check_graph(ops)


def test_a_serial_node_around_a_loop_leaves_the_time_step_outside() -> None:
    """A serial node may wrap the PIMPLE-like loop whole: the time step stays outside."""
    ctx = contexts(["a", "b", "c"], CHAIN, scheme="serial")["a"]
    ctx.time = _Time(1.0)
    ops = _graph(
        ctx,
        _time_loop([1.0], [], [], inner=True),
        _coupling_loop("a", Placement(("inner_loop",))),
    )

    Coupling.of(ctx).check_graph(ops)
    (loop,) = ops.ops
    assert [op.operation_name for op in loop.sub_operations] == [
        "set_time_step",
        "a.time_step",
        "increment_time",
        "a.coupling",
    ]
    assert [op.operation_name for op in loop.sub_operations[3].sub_operations] == ["inner_loop"]
