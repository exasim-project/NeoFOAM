# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Several coupling loops: one op per loop of a region, each with its own leader.

Every port sends a fixed value, so every loop converges after its second pass.
"""

from __future__ import annotations

from functools import partial
from typing import Optional

import pytest
from framework.coupling.regions import (
    Outcomes,
    RecordingCommChannel,
    add_coupled,
    comm_channel,
    contexts,
    couple,
    fixed,
    graph,
    run,
    solve,
    tree,
)

from neofoam.framework.context import Context
from neofoam.framework.coupling import (
    CouplingLoop,
    Exchange,
    InProcessCommChannel,
    coupling,
)
from neofoam.framework.coupling.run import run_participants
from neofoam.framework.execution_observer import observe
from neofoam.framework.graph import DAGResolver
from neofoam.framework.operations import IterativeOp, Operation, StepBuilder
from neofoam.framework.types import OperationMetadata, Placement

X = Exchange("x", (("a", "p"), ("b", "p")))
Y = Exchange("y", (("a", "q"), ("b", "q")))


def _two_loops(comm_channel: Optional[InProcessCommChannel] = None) -> dict[str, Context]:
    """a and b, coupled over x in the outer loop and over y in the inner loop (parallel)."""
    loops = [
        CouplingLoop("outer", ("x",), scheme="parallel"),
        CouplingLoop(
            "inner",
            ("y",),
            placement={"a": Placement(("solve_y",)), "b": Placement(("solve_y",))},
        ),
    ]
    regions = contexts(["a", "b"], [X, Y], loops, comm_channel)
    for name, ctx in regions.items():
        add_coupled(ctx, "p", fixed(1.0, [], data="x"))
        add_coupled(ctx, "q", fixed(2.0, [], data="y"))
    return regions


def test_a_region_gets_one_coupling_op_per_loop_in_loop_order() -> None:
    regions = _two_loops()

    outer, inner = coupling(regions["a"]).operations()

    assert [outer.operation_name, inner.operation_name] == ["a.outer", "a.inner"]
    assert (outer.metadata.wraps, inner.metadata.wraps) == (Placement(), Placement(("solve_y",)))
    assert [port.name for port in couple(regions["a"], 0).ports] == ["p"]
    assert [port.name for port in couple(regions["a"], 1).ports] == ["q"]
    assert couple(regions["a"], 1).loop.name == "inner"


def test_an_inner_loop_with_its_placement_nests_inside_the_outer_loop() -> None:
    regions = _two_loops()

    ops = graph(regions["a"], solve("x", lambda ctx: None), solve("y", lambda ctx: None)).ops

    assert tree(ops) == [("a.outer", ["solve_x", ("a.inner", ["solve_y"])])]


def test_the_messages_of_each_loop_are_tagged_by_loop() -> None:
    comm_channel = RecordingCommChannel()
    regions = _two_loops(comm_channel)

    run(
        {
            name: (ctx, [solve("x", lambda ctx: None), solve("y", lambda ctx: None)])
            for name, ctx in regions.items()
        }
    )

    tags = set(comm_channel.tags)
    assert {"outer/streams", "outer/pass", "outer/decision", "port:x"} <= tags
    assert {"inner/streams", "inner/pass", "inner/decision", "port:y"} <= tags
    assert not tags & {"streams", "pass", "decision"}
    for name in ("outer", "inner"):
        leader = couple(regions["b"], ["outer", "inner"].index(name)).controller
        assert leader is not None and leader.outcome is not None
        assert leader.outcome.converged


def test_each_loop_has_its_own_leader() -> None:
    exchanges = [X, Exchange("z", (("b", "q"), ("c", "q")))]
    # in b, ab around solve_x and bc around solve_z: neither nests in the other
    loops = [
        CouplingLoop("ab", ("x",), placement={"b": Placement(("solve_x",))}),
        CouplingLoop("bc", ("z",), placement={"b": Placement(("solve_z",))}),
    ]
    regions = contexts(["a", "b", "c"], exchanges, loops)
    add_coupled(regions["a"], "p", fixed(1.0, [], data="x"))
    add_coupled(regions["b"], "p", fixed(2.0, [], data="x"))
    add_coupled(regions["b"], "q", fixed(3.0, [], data="z"))
    add_coupled(regions["c"], "q", fixed(4.0, [], data="z"))

    fields = {"a": ["x"], "b": ["x", "z"], "c": ["z"]}
    run(
        {
            name: (ctx, [solve(field, lambda ctx: None) for field in fields[name]])
            for name, ctx in regions.items()
        }
    )

    ops = {
        "a.ab": couple(regions["a"]),
        "b.ab": couple(regions["b"], 0),
        "b.bc": couple(regions["b"], 1),
        "c.bc": couple(regions["c"]),
    }
    assert {label: (op.label, op.group, op.is_leader) for label, op in ops.items()} == {
        "a.ab": ("a.ab", ("a", "b"), False),
        "b.ab": ("b.ab", ("a", "b"), True),
        "b.bc": ("b.bc", ("b", "c"), False),
        "c.bc": ("c.bc", ("b", "c"), True),
    }
    assert [op.controller is not None for op in ops.values()] == [False, True, False, True]
    region_coupling = coupling(regions["b"]).coupling
    assert region_coupling is not None
    assert (region_coupling.group, region_coupling.leader) == (("a", "b", "c"), "c")


def _chain() -> tuple[dict[str, Context], dict[str, list[Operation]]]:
    """a-b-c, ab over x and bc over z: by default bc nests inside ab in b, not in c."""
    exchanges = [X, Exchange("z", (("b", "q"), ("c", "q")))]
    loops = [
        CouplingLoop("ab", ("x",), max_iterations=2),
        CouplingLoop("bc", ("z",)),
    ]
    regions = contexts(["a", "b", "c"], exchanges, loops)
    add_coupled(regions["a"], "p", fixed(1.0, [], data="x"))
    add_coupled(regions["b"], "p", fixed(2.0, [], data="x"))
    add_coupled(regions["b"], "q", fixed(3.0, [], data="z"))
    add_coupled(regions["c"], "q", fixed(4.0, [], data="z"))

    fields = {"a": ["x"], "b": ["x", "z"], "c": ["z"]}
    return regions, {
        name: [solve(field, lambda ctx: None) for field in fields[name]] for name in regions
    }


def test_a_loop_nested_in_one_region_only_raises_at_its_first_pass() -> None:
    # b's second ab pass would run bc again, which c, done with bc, never answers
    regions, ops = _chain()

    with pytest.raises(
        ValueError,
        match=r"coupling 'bc' sits in different loops in its regions: b: ab, c: top; a coupling "
        r"loop must run equally often in all its regions . put the exchanges in one coupling "
        r"loop or place it alike",
    ):
        run({name: (ctx, ops[name]) for name, ctx in regions.items()})


def _counting(name: str, passes: int, *ops: Operation) -> Operation:
    loop = Operation(
        func=IterativeOp(lambda ctx: ctx.loops[-1].iteration < passes),
        metadata=OperationMetadata(op_name=name),
    )
    loop.sub_operations.extend(ops)
    return loop


def _nested(passes: dict[str, tuple[int, int]]) -> dict[str, Context]:
    """a and b, loop ``n`` around solve_x in loops ``outer`` > ``inner`` of ``passes`` each."""
    placement = Placement(("solve_x",))
    loops = [CouplingLoop("n", ("x",), placement={"a": placement, "b": placement})]
    regions = contexts(["a", "b"], [X], loops)
    tasks = {}
    for name, ctx in regions.items():
        add_coupled(ctx, "p", fixed(1.0, [], data="x"))
        outer, inner = passes[name]
        builder = StepBuilder(
            [_counting("outer", outer, _counting("inner", inner, solve("x", lambda ctx: None)))]
        )
        resolved = DAGResolver().resolve(builder, coupling(ctx).operations()).operations
        tasks[name] = partial(resolved.run, ctx)
    run_participants(tasks, comm_channel(regions["a"]))
    return regions


def test_a_loop_placed_alike_runs_in_all_its_regions() -> None:
    outcomes = Outcomes()
    with observe(outcomes):
        regions = _nested({"a": (2, 2), "b": (2, 2)})

    assert couple(regions["b"]).controller is not None and len(outcomes.of("b.n")) == 4


def test_a_loop_entered_in_different_loop_states_raises() -> None:
    # a loop runs four times in both, but a in outer 1 inner 2 where b in outer 2 inner 1
    with pytest.raises(
        ValueError,
        match=r"coupling 'n': its regions entered pass 1 in different loop states: "
        r"a: outer#1 > inner#2, b: outer#2 > inner#1",
    ):
        _nested({"a": (1, 2), "b": (2, 1)})
