# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""``@coupled`` boundary conditions, the exchanges that join them and their coupling ops.

``_Wall`` is a coupled boundary condition of a region that solves ``T``: it sends
its wall temperature (``WallTemperature``) and receives a heat flux
(``WallHeatFlux``), one model per direction. ``_Flux`` is its counterpart. Each
region is a :class:`Context` of its own, coupled as ``system/regionCoupling.yaml``
would couple it, and runs its own graph ``solve_T`` on its own thread.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from framework.coupling.regions import contexts, couple, graph, run, solve
from pydantic import BaseModel

from neofoam.framework.context import Context
from neofoam.framework.coupling import (
    Array,
    Coupling,
    CouplingOp,
    ExchangeSide,
    InProcessTransport,
    Link,
    NearestNeighborMapping,
    RegionCoupling,
    coupled,
    region_couplings,
)
from neofoam.framework.operations import IterativeOp, Operation
from neofoam.framework.types import Placement


class WallTemperature(BaseModel):
    T: Array


class WallHeatFlux(BaseModel):
    q: Array
    h: float


class _Patch:
    def __init__(self, ctx: Context, patch: str, field: str = "T") -> None:
        self.ctx, self.patch, self.field = ctx, patch, field
        self.received: list[Any] = []
        Coupling.of(ctx).add(self)

    @coupled.coordinates
    def points(self) -> Any:
        return np.array([[0.0], [1.0]])


@coupled
class _Wall(_Patch):
    """Sends its temperature, receives a flux."""

    @coupled.sends
    def send(self) -> WallTemperature:
        return WallTemperature(T=self.ctx.fields["T"])

    @coupled.receives
    def receive(self, data: WallHeatFlux) -> None:
        self.received.append(data)


@coupled
class _Flux(_Patch):
    """Sends a flux, receives a temperature."""

    @coupled.sends
    def send(self) -> WallHeatFlux:
        return WallHeatFlux(q=-self.ctx.fields["T"], h=2.0)

    @coupled.receives
    def receive(self, data: WallTemperature) -> None:
        self.received.append(data)


def _region(name: str) -> Context:
    return contexts([name, "other"], [Link("x", ((name, "right"), ("other", "p")))])[name]


CHT = Link("cht", (("solid", "right"), ("fluid", "wall")), "nearest-neighbor")


def _pair(link: Link = CHT) -> dict[str, Context]:
    regions = contexts(["solid", "fluid"], [link], max_iterations=1)
    for ctx in regions.values():
        ctx.fields["T"] = np.array([1.0, 2.0])
    return regions


def _run(regions: dict[str, Context]) -> None:
    run({name: (ctx, [solve("T", lambda ctx: None)]) for name, ctx in regions.items()})


def _tree(ops: list[Operation]) -> list[Any]:
    return [
        (op.operation_name, _tree(op.sub_operations))
        if isinstance(op.func, IterativeOp)
        else op.operation_name
        for op in ops
    ]


def test_coupled_reads_the_models_off_the_annotations() -> None:
    declaration = _Wall.__coupled__  # type: ignore[attr-defined]

    assert (declaration.sends, declaration.receives) == (WallTemperature, WallHeatFlux)
    assert (declaration.send, declaration.receive, declaration.coordinates) == (
        "send",
        "receive",
        "points",
    )


def test_a_coupled_boundary_condition_adds_its_side_to_its_region_s_coupling() -> None:
    region = _region("solid")

    wall = _Wall(region, "right")

    (operation,) = Coupling.of(region).operations()
    assert isinstance(operation.func, CouplingOp)
    assert (operation.operation_name, operation.metadata.wraps) == ("solid.coupling", Placement())
    (side,) = operation.func.sides
    assert (side.region, side.patch, side.field) == ("solid", "right", "T")
    assert (side.sends, side.receives, side.points) == (WallTemperature, WallHeatFlux, 2)
    assert getattr(side.send, "__self__") is wall


def test_coupled_needs_sends_and_receives() -> None:
    with pytest.raises(TypeError, match=r"needs a method marked @coupled.receives"):

        @coupled
        class _OnlySends(_Patch):
            @coupled.sends
            def send(self) -> WallTemperature:
                raise NotImplementedError


def test_coupled_needs_a_pydantic_model_as_annotation() -> None:
    with pytest.raises(
        TypeError, match=r"_NoModel.send \(@coupled.sends\) declares .* got <class 'float'>"
    ):

        @coupled
        class _NoModel(_Patch):
            @coupled.sends
            def send(self) -> float:
                raise NotImplementedError

            @coupled.receives
            def receive(self, data: WallTemperature) -> None:
                raise NotImplementedError


def test_coupled_needs_ctx_and_patch() -> None:
    @coupled
    class _Anonymous:
        def __init__(self, ctx: Context) -> None:
            self.ctx = ctx

        @coupled.sends
        def send(self) -> WallTemperature:
            raise NotImplementedError

        @coupled.receives
        def receive(self, data: WallTemperature) -> None:
            raise NotImplementedError

    with pytest.raises(TypeError, match="sets no self.field in __init__"):
        Coupling.of(_region("a")).add(_Anonymous(_region("a")))


def test_coupled_couples_the_field_of_each_instance() -> None:
    links = [
        Link("a", (("solid", "right"), ("x", "p"))),
        Link("b", (("solid", "left"), ("x", "q"))),
    ]
    region = contexts(["solid", "x"], links)["solid"]

    _Wall(region, "right", field="T")
    _Wall(region, "left", field="U")

    (operation,) = Coupling.of(region).operations()
    assert operation.metadata.wraps == Placement()
    assert [(side.region, side.patch, side.field) for side in couple(region).sides] == [
        ("solid", "right", "T"),
        ("solid", "left", "U"),
    ]


def test_coupled_exchange_with_one_model_per_direction() -> None:
    regions = _pair()
    wall, flux = _Wall(regions["solid"], "right"), _Flux(regions["fluid"], "wall")

    _run(regions)

    (received_flux,) = wall.received
    (received_temperature,) = flux.received
    assert isinstance(received_flux, WallHeatFlux) and isinstance(
        received_temperature, WallTemperature
    )
    assert (received_flux.q.tolist(), received_flux.h) == ([-1.0, -2.0], 2.0)
    assert received_temperature.T.tolist() == [1.0, 2.0]


def test_each_region_gets_its_own_graph_with_its_coupling_op() -> None:
    regions = _pair()
    _Wall(regions["solid"], "right")
    _Flux(regions["fluid"], "wall")

    trees = {
        name: _tree(graph(ctx, solve("T", lambda ctx: None)).ops) for name, ctx in regions.items()
    }

    assert trees == {
        "solid": [("solid.coupling", ["solve_T"])],
        "fluid": [("fluid.coupling", ["solve_T"])],
    }


def test_coupling_op_public_read_api() -> None:
    regions = _pair()
    _Wall(regions["solid"], "right")
    _Flux(regions["fluid"], "wall")
    solid_op, fluid_op = (couple(ctx) for ctx in regions.values())
    assert isinstance(solid_op, CouplingOp) and isinstance(fluid_op, CouplingOp)

    (solid_side,), (fluid_side,) = solid_op.sides, fluid_op.sides
    assert (solid_op.label, solid_op.region, solid_op.is_leader) == (
        "solid.coupling",
        "solid",
        False,
    )
    assert (fluid_op.label, fluid_op.region, fluid_op.is_leader) == (
        "fluid.coupling",
        "fluid",
        True,
    )
    assert [
        (s.region, s.patch, s.field, s.sends, s.receives, s.points, s.mapped)
        for s in (solid_side, fluid_side)
    ] == [
        ("solid", "right", "T", WallTemperature, WallHeatFlux, 2, ("q",)),
        ("fluid", "wall", "T", WallHeatFlux, WallTemperature, 2, ("T",)),
    ]
    assert (solid_op.partner(solid_side), fluid_op.partner(fluid_side)) == (
        "fluid.wall",
        "solid.right",
    )
    exchange = solid_op.exchange_of(solid_side)
    assert (exchange.name, exchange.partner) == ("cht", ("fluid", "wall"))
    assert isinstance(exchange.mapping, NearestNeighborMapping)
    # the leader builds its controller at its first loop test
    assert (solid_op.controller, fluid_op.controller) == (None, None)

    _run(regions)

    controller = fluid_op.controller
    assert solid_op.controller is None and controller is not None
    assert (controller.label, controller.parallel, controller.iteration) == (
        "fluid.coupling",
        False,
        0,
    )
    assert [(end, s.field, s.arrays, s.partner) for end, s in controller.sides.items()] == [
        (("solid", "right"), "T", ("T",), ("fluid", "wall")),
        (("fluid", "wall"), "T", ("q",), ("solid", "right")),
    ]
    # what the sides send does not change: converged after one pass
    assert [(o.passes, o.converged) for o in controller.outcomes] == [(1, True)]


class _Other(BaseModel):
    q: float
    h: float


@coupled
class _Odd(_Patch):
    """Sends a flux whose q is plain, receives a temperature with an extra field."""

    @coupled.sends
    def send(self) -> _Other:
        raise NotImplementedError

    @coupled.receives
    def receive(self, data: WallHeatFlux) -> None:
        raise NotImplementedError


def test_each_region_checks_that_its_side_receives_what_the_partner_sends() -> None:
    regions = _pair(Link("solid-fluid", (("solid", "right"), ("fluid", "wall"))))
    _Wall(regions["solid"], "right")
    _Odd(regions["fluid"], "wall")

    with pytest.raises(ValueError) as error:
        _run(regions)

    # the region that checked first raises; the other learns that it left
    assert str(error.value) in (
        "exchange 'solid-fluid': each side must receive what the other sends: "
        "'T' is in WallTemperature (sent by solid.right) only; "
        "'q' is in WallHeatFlux (received by fluid.wall) only; "
        "'h' is in WallHeatFlux (received by fluid.wall) only",
        "exchange 'solid-fluid': each side must receive what the other sends: "
        "'q' is float in _Other (sent by fluid.wall) but Array in WallHeatFlux "
        "(received by solid.right)",
    )


def test_exchange_with_nearest_neighbor_needs_coordinates() -> None:
    @coupled
    class _Pointless(_Flux):
        @coupled.coordinates
        def points(self) -> Any:
            return None

    regions = _pair()
    _Wall(regions["solid"], "right")
    _Pointless(regions["fluid"], "wall")

    with pytest.raises(ValueError, match="'nearest-neighbor' needs the coordinates of both sides"):
        _run(regions)


@coupled
class _Same(_Patch):
    @coupled.sends
    def send(self) -> WallTemperature:
        return WallTemperature(T=self.ctx.fields["T"])

    @coupled.receives
    def receive(self, data: WallTemperature) -> None:
        self.received.append(data)


def test_a_chain_of_exchanges_is_one_coupling_group() -> None:
    links = [
        Link("a-b", (("a", "to_b"), ("b", "to_a"))),
        Link("b-c", (("b", "to_c"), ("c", "to_b"))),
    ]
    regions = contexts(["a", "b", "c"], links, max_iterations=1)
    for ctx in regions.values():
        ctx.fields["T"] = np.array([1.0, 2.0])
    _Same(regions["a"], "to_b")
    _Same(regions["b"], "to_a")
    _Same(regions["b"], "to_c")
    _Same(regions["c"], "to_b")
    ops = {name: couple(ctx) for name, ctx in regions.items()}

    _run(regions)

    assert [[side.patch for side in op.sides] for op in ops.values()] == [
        ["to_b"],
        ["to_a", "to_c"],
        ["to_b"],
    ]
    assert [op.is_leader for op in ops.values()] == [False, False, True]
    controller = ops["c"].controller
    assert controller is not None
    # serial: a side receives fresh data from a region that steps before its own
    assert {end: controller.fresh(end) for end in controller.sides} == {
        ("a", "to_b"): False,
        ("b", "to_a"): True,
        ("b", "to_c"): False,
        ("c", "to_b"): True,
    }


def _coupling(ctx: Context) -> RegionCoupling:
    coupling = Coupling.of(ctx).coupling
    assert coupling is not None
    return coupling


def test_unconnected_groups_of_exchanges_get_a_leader_each() -> None:
    links = [Link("a-b", (("a", "p"), ("b", "p"))), Link("c-d", (("c", "p"), ("d", "p")))]
    regions = contexts(list("abcd"), links)

    couplings = [_coupling(ctx) for ctx in regions.values()]
    assert [(coupling.group, coupling.leader) for coupling in couplings] == [
        (("a", "b"), "b"),
        (("a", "b"), "b"),
        (("c", "d"), "d"),
        (("c", "d"), "d"),
    ]


@pytest.mark.parametrize(
    ("link", "error"),
    [
        (Link("x", (("a", "p"), ("fluid", "q"))), r"couple the regions \['fluid'\]"),
        (Link("x", (("a", "p"), ("a", "q"))), "exchange 'x' joins two patches of region a"),
    ],
)
def test_region_couplings_check_the_links(link: Link, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        region_couplings(["a", "b"], [link], InProcessTransport())


def test_a_patch_is_in_one_exchange() -> None:
    links = [Link("x", (("a", "p"), ("b", "p"))), Link("y", (("a", "p"), ("b", "q")))]

    with pytest.raises(ValueError, match="a.p is in the exchanges 'x' and 'y'"):
        region_couplings(["a", "b"], links, InProcessTransport())


def test_a_region_no_exchange_names_is_not_coupled() -> None:
    couplings = region_couplings(
        ["a", "b", "c"], [Link("x", (("a", "p"), ("b", "p")))], InProcessTransport()
    )

    assert list(couplings) == ["a", "b"]


class _Film(BaseModel):
    h: float
    phase: str | None


class _Wet(BaseModel):
    q: Array
    film: _Film


@coupled
class _WetFlux(_Patch):
    """Sends a flux with a nested plain model, receives a temperature."""

    @coupled.sends
    def send(self) -> _Wet:
        return _Wet(q=-self.ctx.fields["T"], film=_Film(h=3.0, phase=None))

    @coupled.receives
    def receive(self, data: WallTemperature) -> None:
        self.received.append(data)


@coupled
class _WetWall(_Wall):
    @coupled.receives
    def receive(self, data: _Wet) -> None:
        self.received.append(data)


def test_a_nested_plain_model_travels_as_its_dump() -> None:
    regions = _pair()
    wall = _WetWall(regions["solid"], "right")
    _WetFlux(regions["fluid"], "wall")

    _run(regions)

    assert wall.received[-1].film == _Film(h=3.0, phase=None)


class _Listed(BaseModel):
    q: Array
    weights: list[float]


class _Nested(BaseModel):
    inner: _Listed


@pytest.mark.parametrize(("model", "field"), [(_Listed, "weights"), (_Nested, "inner")])
def test_a_field_that_is_not_array_holds_plain_data(model: type[BaseModel], field: str) -> None:
    with pytest.raises(TypeError, match=f"the field '{field}' of the exchange model"):
        ExchangeSide(
            "solid",
            "right",
            "T",
            sends=model,
            receives=WallTemperature,
            send=lambda: None,
            receive=print,
        )


def test_a_side_keeps_its_coordinates_as_a_float_array() -> None:
    side = ExchangeSide(
        "solid",
        "right",
        "T",
        sends=WallTemperature,
        receives=WallTemperature,
        send=lambda: None,
        receive=print,
        coordinates=[[0, 0, 0], [1, 0, 0]],
    )

    assert isinstance(side.coordinates, np.ndarray) and side.coordinates.dtype == float
    assert side.points == 2
