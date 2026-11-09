# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Two regions ``a`` and ``b`` joined by one exchange, each on its own thread.

``_Side`` stands in for a coupled boundary condition on patch ``p`` together
with the operation that solves its field ``x``: solving sets its value (one more
each pass, or a fixed one), it sends that value (:class:`_Data`) and logs what
it receives. Each region's coupling op wraps its solve; b, the last region, leads.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Any, Literal, Optional

import numpy as np
import pytest
from framework.coupling.regions import Side, add_coupled, contexts, couple, graph, run, solve
from framework.coupling.regions import transport as transport_of
from pydantic import BaseModel, Field

from neofoam.framework.context import Context
from neofoam.framework.coupling import (
    Array,
    BoundMapping,
    Coupling,
    CouplingMapping,
    CouplingNode,
    CouplingNotConvergedWarning,
    InProcessTransport,
    Link,
    RelativeConvergence,
    SideChange,
    TransportClosed,
)
from neofoam.framework.coupling.controller import (
    CouplingController,
    CouplingOutcome,
    GroupSide,
    relative_changes,
)
from neofoam.framework.coupling.data import ExchangeSide
from neofoam.framework.coupling.loop import CouplingOp
from neofoam.framework.coupling.region import region_couplings
from neofoam.framework.execution_observer import ExecutionObserver, observe
from neofoam.framework.types import Placement

LINK = Link("a-b", (("a", "p"), ("b", "p")))


class _Data(BaseModel):
    x: Array


class _Side:
    """A test side of region ``name``: solving counts up (or sets ``fixed``), it sends the value."""

    def __init__(self, name: str, log: list[str], fixed: Optional[float] = None) -> None:
        self.name = name
        self.log = log
        self.fixed = fixed
        self.value = 0.0
        self.solves = 0

    def solve(self, ctx: Context) -> None:
        self.solves += 1
        self.value = self.fixed if self.fixed is not None else self.value + 1
        self.log.append(f"solve_{self.name}")

    def send(self) -> _Data:
        return _Data(x=[self.value])

    def receive(self, data: _Data) -> None:
        self.log.append(f"receive_{self.name} {data.x[0]}")

    def side(self) -> Side:
        return Side(_Data, _Data, self.send, self.receive, np.zeros((1, 1)))


def _never(changes: list[SideChange]) -> bool:
    return False


def _couple(a: _Side, b: _Side, **loop: Any) -> dict[str, CouplingOp]:
    """Couple ``a`` and ``b`` over patch ``p``, run one time step; their coupling ops."""
    regions = contexts(["a", "b"], [LINK], **loop)
    for side in (a, b):
        add_coupled(regions[side.name], "p", "x", side.side())
    run({side.name: (regions[side.name], [solve("x", side.solve)]) for side in (a, b)})
    return {name: couple(ctx) for name, ctx in regions.items()}


def _controller(ops: dict[str, CouplingOp]) -> CouplingController:
    controller = ops["b"].controller
    assert controller is not None
    return controller


@pytest.mark.filterwarnings("ignore::neofoam.framework.coupling.CouplingNotConvergedWarning")
def test_coupling_op_serial_side_receives_what_the_other_sent_in_the_same_pass() -> None:
    log: list[str] = []
    a, b = _Side("a", log), _Side("b", log)

    _couple(a, b, convergence=[_never], max_iterations=2)

    assert log == [
        "receive_a 0.0",
        "solve_a",
        "receive_b 1.0",
        "solve_b",
        "receive_a 1.0",
        "solve_a",
        "receive_b 2.0",
        "solve_b",
    ]


@pytest.mark.filterwarnings("ignore::neofoam.framework.coupling.CouplingNotConvergedWarning")
def test_coupling_op_parallel_sides_receive_what_the_other_sent_in_the_pass_before() -> None:
    a, b = _Side("a", []), _Side("b", [])

    _couple(a, b, convergence=[_never], max_iterations=2, scheme="parallel")

    assert a.log == ["receive_a 0.0", "solve_a", "receive_a 1.0", "solve_a"]
    assert b.log == ["receive_b 0.0", "solve_b", "receive_b 1.0", "solve_b"]


@pytest.mark.parametrize("parallel", [False, True], ids=["serial", "parallel"])
def test_coupling_op_default_controller_takes_the_scheme_of_the_node(parallel: bool) -> None:
    log: list[str] = []
    a, b = _Side("a", log, fixed=1.0), _Side("b", log, fixed=2.0)
    node = CouplingNode("coupling", ("a-b",), scheme="parallel" if parallel else "serial")
    couplings = region_couplings(["a", "b"], [LINK], InProcessTransport(), nodes=[node])
    regions = {
        name: Context(fields={}, models={"coupling": Coupling(couplings[name])}, region=name)
        for name in ("a", "b")
    }
    for side in (a, b):
        add_coupled(regions[side.name], "p", "x", side.side())

    run({side.name: (regions[side.name], [solve("x", side.solve)]) for side in (a, b)})

    controller = _controller({name: couple(ctx) for name, ctx in regions.items()})
    assert controller.parallel is parallel
    assert controller.outcomes[-1].converged
    assert ("receive_b 0.0" in log) is parallel


@pytest.mark.parametrize("parallel", [False, True], ids=["serial", "parallel"])
@pytest.mark.filterwarnings("ignore::neofoam.framework.coupling.CouplingNotConvergedWarning")
def test_coupling_op_sends_fresh_data_as_the_leader_decided_not_by_the_node_scheme(
    parallel: bool,
) -> None:
    # the node's scheme says the opposite of the controller's: the regions follow the leader
    log: list[str] = []
    a, b = _Side("a", log), _Side("b", log)
    node = CouplingNode(
        "coupling",
        ("a-b",),
        scheme="serial" if parallel else "parallel",
        convergence=(_never,),
        max_iterations=2,
    )

    def controller(sides: Any, label: str, /, *, node: CouplingNode) -> CouplingController:
        return CouplingController(
            sides, label, convergence=node.convergence, max_iterations=2, parallel=parallel
        )

    transport = InProcessTransport()
    couplings = region_couplings(["a", "b"], [LINK], transport, nodes=[node], controller=controller)
    regions = {
        name: Context(fields={}, models={"coupling": Coupling(couplings[name])}, region=name)
        for name in ("a", "b")
    }
    for side in (a, b):
        add_coupled(regions[side.name], "p", "x", side.side())
    sent: list[str] = []
    send = transport.send

    def recording(source: str, target: str, tag: str, message: Any) -> None:
        sent.append(tag)
        send(source, target, tag, message)

    transport.send = recording  # type: ignore[method-assign]

    run({side.name: (regions[side.name], [solve("x", side.solve)]) for side in (a, b)})

    if parallel:
        assert [entry for entry in log if entry.endswith("a") or "_a " in entry] == [
            "receive_a 0.0",
            "solve_a",
            "receive_a 1.0",
            "solve_a",
        ]
        assert [entry for entry in log if entry.endswith("b") or "_b " in entry] == [
            "receive_b 0.0",
            "solve_b",
            "receive_b 1.0",
            "solve_b",
        ]
    else:
        assert log == [
            "receive_a 0.0",
            "solve_a",
            "receive_b 1.0",
            "solve_b",
            "receive_a 1.0",
            "solve_a",
            "receive_b 2.0",
            "solve_b",
        ]
    # a sends its data to b once per pass b receives it fresh, never else
    assert sent.count("data:a-b") == (0 if parallel else 2)


@pytest.mark.parametrize(
    ("parallel", "carried"),
    [(False, [[0.0], [2.0]]), (True, [[0.0, 0.0], [1.0, 2.0]])],
    ids=["serial: b only", "parallel: a and b"],
)
@pytest.mark.filterwarnings("ignore::neofoam.framework.coupling.CouplingNotConvergedWarning")
def test_coupling_op_accelerates_the_data_carried_into_a_pass(
    parallel: bool, carried: list[list[float]]
) -> None:
    relaxed: list[list[float]] = []

    def acceleration(sent: list[dict[str, Any]], iteration: int) -> list[dict[str, Any]]:
        relaxed.append([float(data["x"][0]) for data in sent])
        return sent

    a, b = _Side("a", [], fixed=1.0), _Side("b", [], fixed=2.0)
    _couple(
        a,
        b,
        convergence=[_never],
        max_iterations=2,
        scheme="parallel" if parallel else "serial",
        acceleration={"a-b": acceleration},
    )

    assert relaxed == carried


def test_coupling_op_stops_once_the_sent_data_is_unchanged() -> None:
    a, b = _Side("a", [], fixed=5.0), _Side("b", [], fixed=7.0)

    _couple(a, b, max_iterations=10)

    # unchanged after the second pass: one more, the final pass
    assert (a.solves, b.solves) == (3, 3)


@pytest.mark.filterwarnings("ignore::neofoam.framework.coupling.CouplingNotConvergedWarning")
def test_every_region_stops_at_max_iterations() -> None:
    a, b = _Side("a", []), _Side("b", [])

    _couple(a, b, convergence=[_never], max_iterations=3)

    assert (a.solves, b.solves) == (3, 3)


def test_the_leader_records_the_outcome_of_a_converged_loop() -> None:
    ops = _couple(_Side("a", [], fixed=5.0), _Side("b", [], fixed=0.0), max_iterations=10)

    controller = _controller(ops)
    assert controller.outcomes == (CouplingOutcome(passes=3, converged=True, residuals={"x": 0.0}),)
    assert controller.label == "b.coupling"
    assert ops["a"].controller is None


def test_the_leader_warns_and_records_when_max_iterations_ends_the_loop() -> None:
    with pytest.warns(CouplingNotConvergedWarning, match="did not converge in 3 passes"):
        ops = _couple(_Side("a", []), _Side("b", []), convergence=[_never], max_iterations=3)

    # the last pass took both sides from 2 to 3
    assert _controller(ops).outcomes == (
        CouplingOutcome(passes=3, converged=False, residuals={"x": 1 / 3}),
    )


def test_coupling_op_reports_its_outcome_and_each_exchange_to_the_execution_observer() -> None:
    finished: list[tuple[str, CouplingOutcome]] = []
    exchanged: list[tuple[str, str, float]] = []

    class _Observer(ExecutionObserver):
        def coupling_finished(self, coupling: CouplingController, outcome: CouplingOutcome) -> None:
            finished.append((coupling.label, outcome))

        def coupling_exchanged(self, coupling: CouplingOp, side: Any, data: BaseModel) -> None:
            exchanged.append((coupling.label, coupling.partner(side), float(getattr(data, "x")[0])))

    with observe(_Observer()):
        ops = _couple(_Side("a", [], fixed=1.0), _Side("b", [], fixed=2.0), max_iterations=10)

    assert finished == [("b.coupling", _controller(ops).outcomes[0])]
    # serial: a gets b's carried value (0, then 2), b gets a's fresh one
    assert exchanged == [
        ("a.coupling", "b.p", 0.0),
        ("b.coupling", "a.p", 1.0),
        ("a.coupling", "b.p", 2.0),
        ("b.coupling", "a.p", 1.0),
        ("a.coupling", "b.p", 2.0),
        ("b.coupling", "a.p", 1.0),
    ]


def test_relative_changes_is_the_largest_relative_change_per_field() -> None:
    changes = [
        SideChange("T", {"T": np.array([1.0, 0.0])}, {"T": np.array([2.0, 0.0])}),
        SideChange("T", {"q": np.array([0.0])}, {"q": np.array([0.0])}),
        SideChange("U", {"U": np.array([1.0])}, {"U": np.array([0.0])}),
    ]

    assert relative_changes(changes) == {"T": 0.5, "U": 1.0}


def test_coupling_op_wraps_its_region_s_solving_op() -> None:
    regions = contexts(["a", "b"], [LINK])
    add_coupled(regions["a"], "p", "x", _Side("a", []).side())

    (loop,) = graph(regions["a"], solve("x", lambda ctx: None)).ops

    op = loop.func
    assert op is couple(regions["a"]) and isinstance(op, CouplingOp)
    assert (loop.operation_name, loop.metadata.wraps) == ("a.coupling", Placement())
    assert [sub.operation_name for sub in loop.sub_operations] == ["solve_x"]
    (side,) = op.sides
    assert (side.end, side.field, op.partner(side), op.is_leader) == (
        ("a", "p"),
        "x",
        "b.p",
        False,
    )
    exchange = op.exchange_of(side)
    assert (exchange.name, exchange.partner, exchange.mapping.type) == (
        "a-b",
        ("b", "p"),
        "identity",
    )


def test_the_coupling_model_builds_one_coupling_op_once() -> None:
    links = [LINK, Link("a-b-2", (("a", "q"), ("b", "q")))]
    regions = contexts(["a", "b"], links)
    add_coupled(regions["a"], "p", "x", _Side("a", []).side())
    add_coupled(regions["a"], "q", "y", _Side("a", []).side())

    model = Coupling.of(regions["a"])
    (couple_op,) = model.operations()

    assert model.operations() is model.operations()
    assert isinstance(couple_op.func, CouplingOp)
    assert couple_op.metadata.wraps == Placement()
    assert [side.patch for side in couple_op.func.sides] == ["p", "q"]
    with pytest.raises(ValueError, match="a.p is added after the region's graph was built"):
        add_coupled(regions["a"], "p", "x", _Side("a", []).side())


def test_a_region_whose_solver_added_no_coupling_model_cannot_couple() -> None:
    with pytest.raises(ValueError, match="the solver of region a does not support coupling"):
        Coupling.of(Context(fields={}, models={}, region="a"))


def test_a_region_the_case_does_not_couple_has_no_coupling_ops() -> None:
    regions = contexts(["a", "b", "c"], [LINK])

    assert len(Coupling.of(regions["c"]).operations()) == 0
    (loop,) = graph(regions["c"], solve("x", lambda ctx: None)).ops
    assert loop.operation_name == "solve_x"


def test_a_side_in_a_region_the_case_does_not_couple_raises() -> None:
    regions = contexts(["a", "b", "c"], [LINK])

    with pytest.raises(ValueError, match="the case does not couple its region"):
        add_coupled(regions["c"], "p", "x", _Side("c", []).side())


def test_a_coupled_region_without_a_side_raises() -> None:
    regions = contexts(["a", "b"], [LINK])

    with pytest.raises(ValueError, match="exchange 'a-b' connects a.p, which has no coupled side"):
        Coupling.of(regions["a"]).operations()


def test_a_side_on_a_patch_no_exchange_names_raises() -> None:
    regions = contexts(["a", "b"], [LINK])

    with pytest.raises(ValueError, match=r"no exchange connects the coupled side of a\.top"):
        add_coupled(regions["a"], "top", "x", _Side("a", []).side())


def test_two_sides_on_one_patch_raise() -> None:
    regions = contexts(["a", "b"], [LINK])
    add_coupled(regions["a"], "p", "x", _Side("a", []).side())
    add_coupled(regions["a"], "p", "y", _Side("a", []).side())

    with pytest.raises(ValueError, match=r"two coupled sides on patch a\.p"):
        Coupling.of(regions["a"]).operations()


def test_an_exchange_without_a_side_raises() -> None:
    links = [LINK, Link("a-b-2", (("a", "q"), ("b", "q")))]
    regions = contexts(["a", "b"], links)
    add_coupled(regions["a"], "p", "x", _Side("a", []).side())

    with pytest.raises(
        ValueError, match="exchange 'a-b-2' connects a.q, which has no coupled side"
    ):
        Coupling.of(regions["a"]).operations()


def test_only_a_coupled_instance_with_field_and_patch_is_added() -> None:
    model = Coupling.of(contexts(["a", "b"], [LINK])["a"])

    with pytest.raises(TypeError, match="object is not @coupled"):
        model.add(object())


def test_coupling_op_raises_when_send_returns_something_else() -> None:
    a, b = _Side("a", []), _Side("b", [])
    a.send = lambda: {"x": np.zeros(1)}  # type: ignore[assignment, return-value]

    with pytest.raises(
        TypeError, match="side a.p declares it sends _Data, but its send\\(\\) returned dict"
    ):
        _couple(a, b)


def test_a_failing_region_ends_every_region_with_its_error() -> None:
    """b fails in its third solve while a waits for b's data: a must not wait forever."""
    a, b = _Side("a", []), _Side("b", [])
    solve_b = b.solve

    def failing(ctx: Context) -> None:
        if b.solves == 2:
            raise RuntimeError("b failed")
        solve_b(ctx)

    b.solve = failing  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="b failed"):
        _couple(a, b, convergence=[_never], max_iterations=10)

    assert a.solves == 3


def test_a_region_waiting_for_one_that_left_raises() -> None:
    regions = contexts(["a", "b"], [LINK])
    transport = transport_of(regions["a"])
    transport.abort(["b"])

    with pytest.raises(TransportClosed, match="region a waits for 'x' from region b"):
        transport.receive("b", "a", "x")


class _Mixed(BaseModel):
    T: Array
    q: float


class _Reversed(CouplingMapping):
    type: Literal["reversed"] = "reversed"
    mapped: list[dict[str, Any]] = Field(default_factory=list)

    def bind(self, source: Any, target: Any) -> BoundMapping:
        def mapper(data: dict[str, Any]) -> dict[str, Any]:
            self.mapped.append(data)
            return {name: values[::-1] for name, values in data.items()}

        return mapper


@pytest.mark.filterwarnings("ignore::neofoam.framework.coupling.CouplingNotConvergedWarning")
def test_coupling_op_maps_the_arrays_and_hands_plain_values_over_as_they_are() -> None:
    received: list[_Mixed] = []
    mapping = _Reversed()
    regions = contexts(["a", "b"], [Link("a-b", (("a", "p"), ("b", "p")), mapping)])
    for ctx in regions.values():
        side = Side(_Mixed, _Mixed, lambda: _Mixed(T=[1.0, 2.0], q=3.0), received.append)
        add_coupled(ctx, "p", "x", side)

    run({name: (ctx, [solve("x", lambda ctx: None)]) for name, ctx in regions.items()})

    # converged after the first pass, then the final pass
    assert [list(data) for data in mapping.mapped] == [["T"]] * 4
    assert all(isinstance(data, _Mixed) for data in received)
    assert [(data.T.tolist(), data.q) for data in received] == [([2.0, 1.0], 3.0)] * 4


def _live(ctx: Context) -> _Data:
    return _Data(x=ctx.fields["x"])


def test_coupling_op_keeps_a_copy_of_the_data_a_side_sends() -> None:
    """A side that sends its live field and changes it in place still converges by the change."""
    regions = contexts(["a", "b"], [LINK], max_iterations=10)
    for ctx in regions.values():
        ctx.fields["x"] = np.zeros(1)
        add_coupled(ctx, "p", "x", Side(_Data, _Data, partial(_live, ctx), lambda data: None))

    def step(ctx: Context) -> None:
        ctx.fields["x"][0] = min(ctx.fields["x"][0] + 1.0, 3.0)

    run({name: (ctx, [solve("x", step)]) for name, ctx in regions.items()})
    ops = {name: couple(ctx) for name, ctx in regions.items()}

    # x: 1, 2, 3, then 3 again: unchanged after the fourth pass, not after the first;
    # then the final pass
    assert _controller(ops).outcomes == (
        CouplingOutcome(passes=5, converged=True, residuals={"x": 0.0}),
    )


def test_coupled_side_needs_pydantic_models() -> None:
    with pytest.raises(
        TypeError, match="declares what it sends as a pydantic model, got <class 'dict'>"
    ):
        ExchangeSide(
            "a",
            "p",
            "x",
            sends=dict,  # type: ignore[arg-type]
            receives=_Data,
            send=dict,  # type: ignore[arg-type]
            receive=lambda data: None,
        )


class _Tagged(BaseModel):
    x: Array
    tag: str


def test_coupling_op_passes_plain_fields_through_unrelaxed_and_uncompared() -> None:
    """A plain field (here a string) is handed over as sent; relaxation and convergence
    see the Array fields only."""
    received: list[_Tagged] = []
    relaxed: list[list[str]] = []

    def acceleration(sent: list[dict[str, Any]], iteration: int) -> list[dict[str, Any]]:
        relaxed.append(sorted(name for data in sent for name in data))
        return sent

    regions = contexts(["a", "b"], [LINK], max_iterations=10, acceleration={"a-b": acceleration})
    for name, ctx in regions.items():
        side = Side(
            _Tagged, _Tagged, partial(_Tagged, x=[1.0], tag=f"from {name}"), received.append
        )
        add_coupled(ctx, "p", "x", side)

    run({name: (ctx, [solve("x", lambda ctx: None)]) for name, ctx in regions.items()})
    ops = {name: couple(ctx) for name, ctx in regions.items()}

    # what the sides send never changes: converged after one pass, then the final pass
    assert [data.tag for data in received] == ["from b", "from a"] * 2
    assert relaxed == [["x"], ["x"]]
    assert _controller(ops).outcomes == (
        CouplingOutcome(passes=2, converged=True, residuals={"x": 0.0}),
    )


# --- diagnostics: non-finite data, convergence over no data ---------------------------------


def test_a_side_sending_data_that_is_not_finite_raises() -> None:
    a, b = _Side("a", []), _Side("b", [])
    solve_a = a.solve

    def diverging(ctx: Context) -> None:
        solve_a(ctx)
        if a.solves == 2:
            a.value = float("nan")

    a.solve = diverging  # type: ignore[method-assign]

    with pytest.raises(
        FloatingPointError,
        match=r"coupling 'coupling': the x of region a patch p is not finite after pass 2$",
    ):
        _couple(a, b, convergence=[_never], max_iterations=10)


def test_a_relative_change_that_is_not_finite_raises() -> None:
    """The acceleration relaxes b's data to inf: its change in the next pass is not finite."""

    def acceleration(sent: list[dict[str, Any]], iteration: int) -> list[dict[str, Any]]:
        return [{"x": np.full(1, np.inf)} for _data in sent]

    a, b = _Side("a", []), _Side("b", [])

    with pytest.raises(
        FloatingPointError,
        match=r"coupling b.coupling: the relative change of the x of region b patch p is not "
        r"finite in pass 1",
    ):
        _couple(a, b, convergence=[_never], max_iterations=10, acceleration={"a-b": acceleration})


def _group(arrays: tuple[str, ...]) -> list[GroupSide]:
    return [
        GroupSide("a", "p", "x", arrays, ("b", "p"), "a-b"),
        GroupSide("b", "p", "x", arrays, ("a", "p"), "a-b"),
    ]


@pytest.mark.parametrize(
    ("data", "arrays", "error"),
    [
        (
            "x",
            (),
            "coupling b.coupling: convergence RelativeConvergence on x has no Array data to "
            "compare: the sides of x send plain fields only",
        ),
        (
            "T",
            ("x",),
            r"coupling b.coupling: convergence RelativeConvergence on T, which no side of the "
            r"node sends \(its sides couple \['x'\]\)",
        ),
    ],
    ids=["plain fields only", "no side"],
)
def test_a_convergence_over_no_data_raises_as_the_controller_is_built(
    data: str, arrays: tuple[str, ...], error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        CouplingController(
            _group(arrays), "b.coupling", convergence=[RelativeConvergence(data=data, limit=1e-5)]
        )


def test_a_field_reports_an_infinite_residual_until_a_pass_measured_it() -> None:
    controller = CouplingController(_group(("x",)), "b.coupling", convergence=[_never])
    assert controller.residuals == {"x": math.inf}

    # the first decision starts the first pass: nothing measured yet
    controller.decide({("a", "p"): {"x": np.ones(1)}, ("b", "p"): {"x": np.ones(1)}})
    assert controller.residuals == {"x": math.inf}

    controller.decide({("a", "p"): {"x": np.ones(1)}, ("b", "p"): {"x": np.full(1, 2.0)}})
    assert controller.residuals == {"x": 0.5}


def test_relative_changes_keep_a_nan() -> None:
    changes = [
        SideChange("x", {"x": np.ones(1)}, {"x": np.full(1, 2.0)}),
        SideChange("x", {"x": np.ones(1)}, {"x": np.full(1, np.nan)}),
    ]

    assert math.isnan(relative_changes(changes)["x"])
