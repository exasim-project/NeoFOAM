# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""A region's coupling op: two regions ``a`` and ``b``, one exchange, b leads.

``_CoupledInstance`` is a coupled boundary condition with the solve of its field: each
solve counts its value up (or sets a fixed one); it logs what it receives.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Literal, Optional

import numpy as np
import pytest
from framework.coupling.regions import (
    CoupledInstance,
    Outcomes,
    RecordingCommChannel,
    add_coupled,
    context,
    contexts,
    couple,
    graph,
    never,
    run,
    solve,
)
from framework.coupling.regions import comm_channel as comm_channel_of
from pydantic import BaseModel, Field

from neofoam.framework.context import Context
from neofoam.framework.coupling import (
    Array,
    BoundMapping,
    CommChannelClosed,
    CouplingLoop,
    CouplingMapping,
    CouplingNotConvergedWarning,
    Exchange,
    FaceData,
    IdentityMapping,
    InProcessCommChannel,
    coupling,
)
from neofoam.framework.coupling.exchanges import participant_exchanges
from neofoam.framework.coupling.loop._leader import (
    CouplingOutcome,
    Leader,
)
from neofoam.framework.coupling.loop._loop_op import CouplingLoopOp
from neofoam.framework.execution_observer import ExecutionObserver, observe
from neofoam.framework.types import Placement

EXCHANGE = Exchange("a-b", (("a", "p"), ("b", "p")), IdentityMapping())


class _Data(FaceData):
    x: Array


class _CoupledInstance:
    """An instance of region ``name``: solving counts up (or sets ``fixed``), sends the value."""

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

    def instance(self) -> CoupledInstance:
        return CoupledInstance(_Data, _Data, self.send, self.receive, np.zeros((1, 1)))


def _couple(a: _CoupledInstance, b: _CoupledInstance, **loop: Any) -> dict[str, CouplingLoopOp]:
    """Couple ``a`` and ``b`` over patch ``p``, run one time step; their coupling ops."""
    regions = contexts(["a", "b"], [EXCHANGE], **loop)
    for instance in (a, b):
        add_coupled(regions[instance.name], "p", instance.instance())
    run(
        {
            instance.name: (regions[instance.name], [solve("x", instance.solve)])
            for instance in (a, b)
        }
    )
    return {name: couple(ctx) for name, ctx in regions.items()}


def _leader(ops: dict[str, CouplingLoopOp]) -> Leader:
    leader = ops["b"].controller
    assert leader is not None
    return leader


@pytest.mark.filterwarnings("ignore::neofoam.framework.coupling.CouplingNotConvergedWarning")
def test_coupling_op_serial_port_receives_what_the_other_sent_in_the_same_pass() -> None:
    log: list[str] = []
    a, b = _CoupledInstance("a", log), _CoupledInstance("b", log)

    _couple(a, b, convergence=[never], max_iterations=2)

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
def test_coupling_op_parallel_ports_receive_what_the_other_sent_in_the_pass_before() -> None:
    a, b = _CoupledInstance("a", []), _CoupledInstance("b", [])

    _couple(a, b, convergence=[never], max_iterations=2, scheme="parallel")

    assert a.log == ["receive_a 0.0", "solve_a", "receive_a 1.0", "solve_a"]
    assert b.log == ["receive_b 0.0", "solve_b", "receive_b 1.0", "solve_b"]


@pytest.mark.parametrize("parallel", [False, True], ids=["serial", "parallel"])
def test_coupling_op_default_leader_takes_the_scheme_of_the_loop(parallel: bool) -> None:
    log: list[str] = []
    a, b = _CoupledInstance("a", log, fixed=1.0), _CoupledInstance("b", log, fixed=2.0)
    loop = CouplingLoop("coupling", ("a-b",), scheme="parallel" if parallel else "serial")
    couplings = participant_exchanges(["a", "b"], [EXCHANGE], InProcessCommChannel(), loops=[loop])
    regions = {name: context(name, couplings[name]) for name in ("a", "b")}
    for instance in (a, b):
        add_coupled(regions[instance.name], "p", instance.instance())

    run(
        {
            instance.name: (regions[instance.name], [solve("x", instance.solve)])
            for instance in (a, b)
        }
    )

    leader = _leader({name: couple(ctx) for name, ctx in regions.items()})
    # serial: b receives what a, stepping first, sent in this pass; parallel: neither fresh
    assert [leader.fresh(stream) for stream in leader.streams] == [not parallel, False]
    assert not hasattr(leader, "parallel")
    assert leader.outcome is not None and leader.outcome.converged
    assert ("receive_b 0.0" in log) is parallel


@pytest.mark.parametrize(("scheme", "sent"), [("serial", 2), ("parallel", 0)])
@pytest.mark.filterwarnings("ignore::neofoam.framework.coupling.CouplingNotConvergedWarning")
def test_coupling_op_sends_data_only_where_the_partner_receives_it_fresh(
    scheme: Literal["serial", "parallel"], sent: int
) -> None:
    comm_channel = RecordingCommChannel()
    loops = [
        CouplingLoop("coupling", ("a-b",), scheme=scheme, convergence=(never,), max_iterations=2)
    ]
    couplings = participant_exchanges(["a", "b"], [EXCHANGE], comm_channel, loops=loops)
    regions = {name: context(name, couplings[name]) for name in ("a", "b")}
    a, b = _CoupledInstance("a", []), _CoupledInstance("b", [])
    for instance in (a, b):
        add_coupled(regions[instance.name], "p", instance.instance())

    run(
        {
            instance.name: (regions[instance.name], [solve("x", instance.solve)])
            for instance in (a, b)
        }
    )

    assert comm_channel.tags.count("data:a-b/x") == sent


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

    a, b = _CoupledInstance("a", [], fixed=1.0), _CoupledInstance("b", [], fixed=2.0)
    _couple(
        a,
        b,
        convergence=[never],
        max_iterations=2,
        scheme="parallel" if parallel else "serial",
        acceleration={"a-b": acceleration},
    )

    assert relaxed == carried


def test_coupling_op_stops_once_the_sent_data_is_unchanged() -> None:
    a, b = _CoupledInstance("a", [], fixed=5.0), _CoupledInstance("b", [], fixed=7.0)

    _couple(a, b, max_iterations=10)

    # unchanged after the second pass: one more, the final pass
    assert (a.solves, b.solves) == (3, 3)


@pytest.mark.filterwarnings("ignore::neofoam.framework.coupling.CouplingNotConvergedWarning")
def test_every_region_stops_at_max_iterations() -> None:
    a, b = _CoupledInstance("a", []), _CoupledInstance("b", [])

    _couple(a, b, convergence=[never], max_iterations=3)

    assert (a.solves, b.solves) == (3, 3)


def test_the_leader_records_the_outcome_of_a_converged_loop() -> None:
    outcomes = Outcomes()
    with observe(outcomes):
        ops = _couple(
            _CoupledInstance("a", [], fixed=5.0),
            _CoupledInstance("b", [], fixed=0.0),
            max_iterations=10,
        )

    leader = _leader(ops)
    assert outcomes.of("b.coupling") == (
        CouplingOutcome(passes=3, converged=True, residuals={"x": 0.0}),
    )
    assert leader.label == "b.coupling"
    assert ops["a"].controller is None


def test_the_leader_warns_and_records_when_max_iterations_ends_the_loop() -> None:
    outcomes = Outcomes()
    with observe(outcomes):
        with pytest.warns(CouplingNotConvergedWarning, match="did not converge in 3 passes"):
            ops = _couple(
                _CoupledInstance("a", []),
                _CoupledInstance("b", []),
                convergence=[never],
                max_iterations=3,
            )

    # the last pass took both ports from 2 to 3
    assert outcomes.of(_leader(ops).label) == (
        CouplingOutcome(passes=3, converged=False, residuals={"x": 1 / 3}),
    )


def test_coupling_op_reports_its_outcome_and_each_exchange_to_the_execution_observer() -> None:
    finished: list[tuple[str, CouplingOutcome]] = []
    exchanged: list[tuple[str, str, float]] = []

    class _Observer(ExecutionObserver):
        def coupling_finished(self, coupling: Leader, outcome: CouplingOutcome) -> None:
            finished.append((coupling.label, outcome))

        def coupling_exchanged(self, coupling: CouplingLoopOp, port: Any, data: BaseModel) -> None:
            exchanged.append((coupling.label, coupling.partner(port), float(getattr(data, "x")[0])))

    with observe(_Observer()):
        ops = _couple(
            _CoupledInstance("a", [], fixed=1.0),
            _CoupledInstance("b", [], fixed=2.0),
            max_iterations=10,
        )

    assert finished == [("b.coupling", _leader(ops).outcome)]
    # serial: a gets b's carried value (0, then 2), b gets a's fresh one
    assert exchanged == [
        ("a.coupling", "b.p", 0.0),
        ("b.coupling", "a.p", 1.0),
        ("a.coupling", "b.p", 2.0),
        ("b.coupling", "a.p", 1.0),
        ("a.coupling", "b.p", 2.0),
        ("b.coupling", "a.p", 1.0),
    ]


def test_coupling_op_wraps_its_region_s_solving_op() -> None:
    regions = contexts(["a", "b"], [EXCHANGE])
    add_coupled(regions["a"], "p", _CoupledInstance("a", []).instance())

    (loop,) = graph(regions["a"], solve("x", lambda ctx: None)).ops

    op = loop.func
    assert op is couple(regions["a"]) and isinstance(op, CouplingLoopOp)
    assert (loop.operation_name, loop.metadata.wraps) == ("a.coupling", Placement())
    assert [sub.operation_name for sub in loop.sub_operations] == ["solve_x"]
    (port,) = op.ports
    assert (port.end, port.names, op.partner(port), op.is_leader) == (
        ("a", "p"),
        {"x"},
        "b.p",
        False,
    )
    assert op.exchange_of(port) is EXCHANGE


def test_a_coupled_region_without_a_port_raises() -> None:
    regions = contexts(["a", "b"], [EXCHANGE])

    with pytest.raises(ValueError, match="exchange 'a-b' connects a.p, which has no coupled inst"):
        coupling(regions["a"]).operations()


def test_two_instances_on_one_patch_send_a_data_field_once() -> None:
    regions = contexts(["a", "b"], [EXCHANGE])
    add_coupled(regions["a"], "p", _CoupledInstance("a", []).instance())

    with pytest.raises(ValueError, match=r"two @coupled.sends of x on a\.p"):
        add_coupled(regions["a"], "p", _CoupledInstance("a", []).instance())


def test_an_exchange_without_a_port_raises() -> None:
    exchanges = [EXCHANGE, Exchange("a-b-2", (("a", "q"), ("b", "q")), IdentityMapping())]
    regions = contexts(["a", "b"], exchanges)
    add_coupled(regions["a"], "p", _CoupledInstance("a", []).instance())

    with pytest.raises(
        ValueError, match="exchange 'a-b-2' connects a.q, which has no coupled instance"
    ):
        coupling(regions["a"]).operations()


def test_coupling_op_raises_when_send_returns_something_else() -> None:
    a, b = _CoupledInstance("a", []), _CoupledInstance("b", [])
    a.send = lambda: {"x": np.zeros(1)}  # type: ignore[assignment, return-value]

    with pytest.raises(
        TypeError, match="a.p declares it sends _Data, but its @coupled.sends returned dict"
    ):
        _couple(a, b)


def test_a_failing_region_ends_every_region_with_its_error() -> None:
    """b fails in its third solve while a waits for b's data: a must not wait forever."""
    a, b = _CoupledInstance("a", []), _CoupledInstance("b", [])
    solve_b = b.solve

    def failing(ctx: Context) -> None:
        if b.solves == 2:
            raise RuntimeError("b failed")
        solve_b(ctx)

    b.solve = failing  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="b failed"):
        _couple(a, b, convergence=[never], max_iterations=10)

    assert a.solves == 3


def test_a_region_waiting_for_one_that_left_raises() -> None:
    regions = contexts(["a", "b"], [EXCHANGE])
    comm_channel = comm_channel_of(regions["a"])
    comm_channel.abort(["b"])

    with pytest.raises(CommChannelClosed, match="region a waits for 'x' from region b"):
        comm_channel.receive("b", "a", "x")


class _Reversed(CouplingMapping):
    type: Literal["reversed"] = "reversed"
    mapped: list[dict[str, Any]] = Field(default_factory=list)

    def bind(self, source: Any, target: Any) -> BoundMapping:
        def mapper(data: dict[str, Any]) -> dict[str, Any]:
            self.mapped.append(data)
            return {name: values[::-1] for name, values in data.items()}

        return mapper


def test_coupling_op_maps_what_the_partner_sent_onto_the_port_s_points() -> None:
    received: list[_Data] = []
    mapping = _Reversed()
    regions = contexts(["a", "b"], [Exchange("a-b", (("a", "p"), ("b", "p")), mapping)])
    for ctx in regions.values():
        add_coupled(
            ctx, "p", CoupledInstance(_Data, _Data, lambda: _Data(x=[1.0, 2.0]), received.append)
        )

    run({name: (ctx, [solve("x", lambda ctx: None)]) for name, ctx in regions.items()})

    # converged after the first pass, then the final pass
    assert len(mapping.mapped) == 4
    assert [data.x.tolist() for data in received] == [[2.0, 1.0]] * 4


def _live(ctx: Context) -> _Data:
    return _Data(x=ctx.fields["x"])


def test_coupling_op_keeps_a_copy_of_the_data_an_instance_sends() -> None:
    """An instance sending its live field, changed in place, still converges by the change."""
    regions = contexts(["a", "b"], [EXCHANGE], max_iterations=10)
    for ctx in regions.values():
        ctx.fields["x"] = np.zeros(1)
        add_coupled(ctx, "p", CoupledInstance(_Data, _Data, partial(_live, ctx), lambda data: None))

    def step(ctx: Context) -> None:
        ctx.fields["x"][0] = min(ctx.fields["x"][0] + 1.0, 3.0)

    outcomes = Outcomes()
    with observe(outcomes):
        run({name: (ctx, [solve("x", step)]) for name, ctx in regions.items()})
    ops = {name: couple(ctx) for name, ctx in regions.items()}

    # x: 1, 2, 3, 3: unchanged after the fourth pass, then the final pass
    assert outcomes.of(_leader(ops).label) == (
        CouplingOutcome(passes=5, converged=True, residuals={"x": 0.0}),
    )


def test_a_port_sending_data_that_is_not_finite_raises() -> None:
    a, b = _CoupledInstance("a", []), _CoupledInstance("b", [])
    solve_a = a.solve

    def diverging(ctx: Context) -> None:
        solve_a(ctx)
        if a.solves == 2:
            a.value = float("nan")

    a.solve = diverging  # type: ignore[method-assign]

    with pytest.raises(
        FloatingPointError,
        match=r"coupling 'coupling': the x of a.p is not finite after pass 2$",
    ):
        _couple(a, b, convergence=[never], max_iterations=10)


def test_a_relative_change_that_is_not_finite_raises() -> None:
    """The acceleration relaxes b's data to inf: its change in the next pass is not finite."""

    def acceleration(sent: list[dict[str, Any]], iteration: int) -> list[dict[str, Any]]:
        return [{"x": np.full(1, np.inf)} for _data in sent]

    a, b = _CoupledInstance("a", []), _CoupledInstance("b", [])

    with pytest.raises(
        FloatingPointError,
        match=r"coupling b.coupling: the relative change of the x of b.p is not finite in pass 1",
    ):
        _couple(a, b, convergence=[never], max_iterations=10, acceleration={"a-b": acceleration})
