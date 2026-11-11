# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Coupling nodes: one coupling op per node of a region, each with its own leader and strategies.

Regions ``a``, ``b`` (and ``c``) are joined by exchanges on patches ``p`` and
``q``; a ``_Side`` on a patch sends a fixed value of its field and logs what it
receives, so every node converges after its second pass.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Optional

import numpy as np
import pytest
from framework.coupling.regions import (
    RecordingTransport,
    Side,
    add_coupled,
    contexts,
    couple,
    graph,
    run,
    solve,
    transport,
    tree,
)
from pydantic import BaseModel

from neofoam.framework.context import Context
from neofoam.framework.coupling import (
    Array,
    Coupling,
    CouplingNode,
    InProcessTransport,
    Link,
    SideChange,
)
from neofoam.framework.coupling.controller import CouplingController, GroupSide
from neofoam.framework.coupling.region import region_couplings, run_regions
from neofoam.framework.graph import DAGResolver
from neofoam.framework.operations import IterativeOp, Operation, StepBuilder
from neofoam.framework.types import OperationMetadata, Placement

X = Link("x", (("a", "p"), ("b", "p")))
Y = Link("y", (("a", "q"), ("b", "q")))


class _Data(BaseModel):
    v: Array


def _side(value: float, log: list[float]) -> Side:
    return Side(
        _Data,
        _Data,
        lambda: _Data(v=[value]),
        lambda d: log.append(float(d.v[0])),
        np.zeros((1, 1)),
    )


def _two_nodes(transport: Optional[InProcessTransport] = None) -> dict[str, Context]:
    """a and b, coupled over x in the outer node and over y in the inner node (parallel)."""
    nodes = [
        CouplingNode("outer", ("x",), scheme="parallel"),
        CouplingNode(
            "inner",
            ("y",),
            placement={"a": Placement(("solve_y",)), "b": Placement(("solve_y",))},
        ),
    ]
    regions = contexts(["a", "b"], [X, Y], nodes, transport)
    for name, ctx in regions.items():
        add_coupled(ctx, "p", "x", _side(1.0, []))
        add_coupled(ctx, "q", "y", _side(2.0, []))
    return regions


def test_a_region_gets_one_coupling_op_per_node_in_node_order() -> None:
    regions = _two_nodes()

    outer, inner = Coupling.of(regions["a"]).operations()

    assert [outer.operation_name, inner.operation_name] == ["a.outer", "a.inner"]
    assert (outer.metadata.wraps, inner.metadata.wraps) == (Placement(), Placement(("solve_y",)))
    assert [s.patch for s in couple(regions["a"], 0).sides] == ["p"]
    assert [s.patch for s in couple(regions["a"], 1).sides] == ["q"]
    assert couple(regions["a"], 1).node.name == "inner"


def test_an_inner_node_with_its_placement_nests_inside_the_outer_node() -> None:
    regions = _two_nodes()

    ops = graph(regions["a"], solve("x", lambda ctx: None), solve("y", lambda ctx: None)).ops

    assert tree(ops) == [("a.outer", ["solve_x", ("a.inner", ["solve_y"])])]


def test_the_messages_of_each_node_are_tagged_by_node() -> None:
    transport = RecordingTransport()
    regions = _two_nodes(transport)

    run(
        {
            name: (ctx, [solve("x", lambda ctx: None), solve("y", lambda ctx: None)])
            for name, ctx in regions.items()
        }
    )

    tags = set(transport.tags)
    assert {"outer/sides", "outer/pass", "outer/decision", "side:x"} <= tags
    assert {"inner/sides", "inner/pass", "inner/decision", "side:y"} <= tags
    assert not tags & {"sides", "pass", "decision"}
    for name in ("outer", "inner"):
        controller = couple(regions["b"], ["outer", "inner"].index(name)).controller
        assert controller is not None and controller.outcomes[-1].converged


def test_each_node_has_its_own_leader() -> None:
    links = [X, Link("z", (("b", "q"), ("c", "q")))]
    # in b, ab around solve_x and bc around solve_z: neither nests in the other
    nodes = [
        CouplingNode("ab", ("x",), placement={"b": Placement(("solve_x",))}),
        CouplingNode("bc", ("z",), placement={"b": Placement(("solve_z",))}),
    ]
    regions = contexts(["a", "b", "c"], links, nodes)
    add_coupled(regions["a"], "p", "x", _side(1.0, []))
    add_coupled(regions["b"], "p", "x", _side(2.0, []))
    add_coupled(regions["b"], "q", "z", _side(3.0, []))
    add_coupled(regions["c"], "q", "z", _side(4.0, []))

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
    # the group agrees as one: its last region leads it
    coupling = Coupling.of(regions["b"]).coupling
    assert coupling is not None and (coupling.group, coupling.leader) == (("a", "b", "c"), "c")


def _chain() -> tuple[dict[str, Context], dict[str, list[Operation]]]:
    """a-b-c, ab over x and bc over z: by default bc nests inside ab in b, not in c."""
    links = [X, Link("z", (("b", "q"), ("c", "q")))]
    nodes = [
        CouplingNode("ab", ("x",), max_iterations=2),
        CouplingNode("bc", ("z",)),
    ]
    regions = contexts(["a", "b", "c"], links, nodes)
    add_coupled(regions["a"], "p", "x", _side(1.0, []))
    add_coupled(regions["b"], "p", "x", _side(2.0, []))
    add_coupled(regions["b"], "q", "z", _side(3.0, []))
    add_coupled(regions["c"], "q", "z", _side(4.0, []))

    fields = {"a": ["x"], "b": ["x", "z"], "c": ["z"]}
    return regions, {
        name: [solve(field, lambda ctx: None) for field in fields[name]] for name in regions
    }


def test_a_node_nested_in_one_region_only_raises_at_its_first_pass() -> None:
    # b's second ab pass would run bc again, which c, done with bc, never answers
    regions, ops = _chain()

    with pytest.raises(
        ValueError,
        match=r"coupling 'bc' sits in different loops in its regions: b: ab, c: top; a node "
        r"must run equally often in all its regions . put the exchanges in one node or "
        r"place it alike",
    ):
        run({name: (ctx, ops[name]) for name, ctx in regions.items()})


def _counting(name: str, passes: int, *ops: Operation) -> Operation:
    """A loop ``name`` of ``passes`` passes around ``ops``."""
    loop = Operation(
        func=IterativeOp(lambda ctx: ctx.loops[-1].iteration < passes),
        metadata=OperationMetadata(op_name=name),
    )
    loop.sub_operations.extend(ops)
    return loop


def _nested(passes: dict[str, tuple[int, int]]) -> dict[str, Context]:
    """a and b, each node ``n`` around solve_x inside loop ``inner`` inside loop ``outer``.

    ``passes`` are the passes of each region's (outer, inner) loops.
    """
    placement = Placement(("solve_x",))
    nodes = [CouplingNode("n", ("x",), placement={"a": placement, "b": placement})]
    regions = contexts(["a", "b"], [X], nodes)
    tasks = {}
    for name, ctx in regions.items():
        add_coupled(ctx, "p", "x", _side(1.0, []))
        outer, inner = passes[name]
        builder = StepBuilder(
            [_counting("outer", outer, _counting("inner", inner, solve("x", lambda ctx: None)))]
        )
        resolved = DAGResolver().resolve(builder, Coupling.of(ctx).operations()).operations
        tasks[name] = partial(resolved.run, ctx)
    run_regions(tasks, transport(regions["a"]))
    return regions


def test_a_node_placed_alike_runs_in_all_its_regions() -> None:
    regions = _nested({"a": (2, 2), "b": (2, 2)})

    controller = couple(regions["b"]).controller
    assert controller is not None and len(controller.outcomes) == 4


def test_a_node_entered_in_different_loop_states_raises() -> None:
    # a node runs four times in both, but a in outer 1 inner 2 where b in outer 2 inner 1
    with pytest.raises(
        ValueError,
        match=r"coupling 'n': its regions entered pass 1 in different loop states: "
        r"a: outer#1 > inner#2, b: outer#2 > inner#1",
    ):
        _nested({"a": (1, 2), "b": (2, 1)})


def test_a_serial_node_enclosing_a_node_of_two_of_its_regions_raises() -> None:
    # a serial outer pass of b waits for a's whole pass, which waits in its inner
    # node for b
    nodes = [CouplingNode("fsi", ("x",), scheme="serial"), CouplingNode("cht", ("y",))]

    with pytest.raises(
        ValueError,
        match=r"coupling 'fsi' is serial but encloses 'cht', which couples \['a', 'b'\] as "
        r"well; .* make 'fsi' parallel",
    ):
        region_couplings(["a", "b"], [X, Y], InProcessTransport(), nodes=nodes)


def test_a_serial_node_may_enclose_a_node_sharing_one_region() -> None:
    links = [X, Link("z", (("b", "q"), ("c", "q")))]
    nodes = [CouplingNode("ab", ("x",), scheme="serial"), CouplingNode("bc", ("z",))]

    couplings = region_couplings(["a", "b", "c"], links, InProcessTransport(), nodes=nodes)

    assert [node.scheme for node in couplings["b"].nodes] == ["serial", "serial"]


def test_a_node_is_serial_by_default() -> None:
    couplings = region_couplings(["a", "b"], [X, Y], InProcessTransport())

    (node,) = couplings["a"].nodes
    assert (node.name, node.scheme, node.parallel) == ("coupling", "serial", False)


def _sides() -> list[GroupSide]:
    """Two exchanges x and y between a and b."""
    return [
        GroupSide("a", "p", "x", ("b", "p"), "x"),
        GroupSide("b", "p", "x", ("a", "p"), "x"),
        GroupSide("a", "q", "y", ("b", "q"), "y"),
        GroupSide("b", "q", "y", ("a", "q"), "y"),
    ]


def _data(*values: float) -> dict[tuple[str, str], dict[str, Any]]:
    """The data of the sides of :func:`_sides`, in their order."""
    return {side.end: {"v": np.array([value])} for side, value in zip(_sides(), values)}


def test_an_acceleration_relaxes_only_the_sides_of_its_exchange() -> None:
    relaxed: list[list[float]] = []

    def halve(sent: list[dict[str, Any]], iteration: int) -> list[dict[str, Any]]:
        relaxed.append([float(data["v"][0]) for data in sent])
        return [{"v": data["v"] / 2} for data in sent]

    node = CouplingNode("n", ("x", "y"), scheme="parallel", acceleration={"x": halve})
    controller = CouplingController(_sides(), "b.n", node)

    carried = controller.decide(_data(1.0, 2.0, 10.0, 20.0), 0)

    assert carried is not None
    assert relaxed == [[1.0, 2.0]]
    assert [float(data["v"][0]) for data in carried.values()] == [0.5, 1.0, 10.0, 20.0]


def test_a_node_converges_once_every_measure_of_its_convergence_holds() -> None:
    verdicts = {"first": True, "second": False}

    def measure(name: str) -> Any:
        def converged(changes: list[SideChange]) -> bool:
            return verdicts[name]

        return converged

    node = CouplingNode("n", ("x", "y"), convergence=(measure("first"), measure("second")))
    controller = CouplingController(_sides(), "b.n", node)
    assert controller.decide(_data(1.0, 2.0, 3.0, 4.0), 0) is not None
    assert controller.decide(_data(1.0, 2.0, 3.0, 4.0), 1) is not None

    verdicts["second"] = True

    # converged: the final pass, then the end
    assert controller.decide(_data(1.0, 2.0, 3.0, 4.0), 2) is not None
    assert controller.final
    assert controller.decide(_data(1.0, 2.0, 3.0, 4.0), 3) is None
    assert controller.outcomes[-1].converged


def test_a_node_without_convergence_converges_once_the_data_is_unchanged() -> None:
    controller = CouplingController(_sides(), "b.n", CouplingNode("n", ("x", "y")))

    assert controller.decide(_data(1.0, 2.0, 3.0, 4.0), 0) is not None
    assert controller.decide(_data(1.0, 2.0, 3.0, 5.0), 1) is not None
    assert controller.decide(_data(1.0, 2.0, 3.0, 5.0), 2) is not None
    assert controller.final
    assert controller.decide(_data(1.0, 2.0, 3.0, 5.0), 3) is None


def test_the_default_node_holds_every_exchange() -> None:
    couplings = region_couplings(["a", "b"], [X, Y], InProcessTransport())

    (node,) = couplings["a"].nodes
    assert (node.name, node.exchanges, node.placement_of("a")) == (
        "coupling",
        ("x", "y"),
        Placement(),
    )


@pytest.mark.parametrize(
    ("nodes", "error"),
    [
        (
            [CouplingNode("n", ("x",)), CouplingNode("n", ("y",))],
            "two coupling nodes are named 'n'",
        ),
        (
            [CouplingNode("n", ("x", "y")), CouplingNode("m", ("y",))],
            "exchange 'y' is in the coupling nodes 'n' and 'm'",
        ),
        ([CouplingNode("n", ("x",))], r"the exchanges \['y'\] are in no coupling node"),
        (
            [CouplingNode("n", ("x", "w"))],
            "coupling node 'n' names the exchange 'w', which does not exist",
        ),
        (
            [CouplingNode("n", ("x", "y"), placement={"c": Placement()})],
            r"coupling node 'n' places its op in the regions \['c'\], which it does not couple",
        ),
        (
            [
                CouplingNode("n", ("x",), acceleration={"y": lambda s, i: s}),
                CouplingNode("m", ("y",)),
            ],
            r"coupling node 'n' accelerates the exchanges \['y'\], which are not its own",
        ),
    ],
    ids=[
        "duplicate node name",
        "exchange in two nodes",
        "exchange in no node",
        "unknown exchange",
        "unknown region in placement",
        "acceleration of another exchange",
    ],
)
def test_region_couplings_check_the_nodes(nodes: list[CouplingNode], error: str) -> None:
    with pytest.raises(ValueError, match=error):
        region_couplings(["a", "b", "c"], [X, Y], InProcessTransport(), nodes=nodes)


def test_a_region_s_field_is_in_at_most_one_node() -> None:
    nodes = [CouplingNode("n", ("x",), scheme="parallel"), CouplingNode("m", ("y",))]
    regions = contexts(["a", "b"], [X, Y], nodes)
    add_coupled(regions["a"], "p", "T", _side(1.0, []))
    add_coupled(regions["a"], "q", "T", _side(1.0, []))

    with pytest.raises(ValueError, match="field T of region a is coupled in the nodes 'n' and 'm'"):
        Coupling.of(regions["a"]).operations()


def test_two_exchanges_of_one_name_raise() -> None:
    twin = Link("x", (("a", "q"), ("b", "q")))

    with pytest.raises(
        ValueError, match="two exchanges are named 'x'; an exchange's name is unique"
    ):
        region_couplings(
            ["a", "b"], [X, twin], InProcessTransport(), nodes=[CouplingNode("n", ("x",))]
        )
