# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""``@coupled`` boundary conditions and the exchanges that join them.

``_Wall`` sends its wall temperature and receives a heat flux, ``_Flux`` the
reverse: one model per direction, each region on its own thread.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from framework.coupling.regions import contexts, couple, graph, run, solve, tree
from pydantic import BaseModel

from neofoam.framework.context import Context
from neofoam.framework.coupling import (
    Array,
    Coupling,
    Link,
    NearestNeighborMapping,
    coupled,
)
from neofoam.framework.coupling.loop import CouplingOp
from neofoam.framework.types import Placement


class WallTemperature(BaseModel):
    T: Array


class WallHeatFlux(BaseModel):
    q: Array


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
        return WallHeatFlux(q=-self.ctx.fields["T"])

    @coupled.receives
    def receive(self, data: WallTemperature) -> None:
        self.received.append(data)


def _region(name: str) -> Context:
    return contexts([name, "other"], [Link("x", ((name, "right"), ("other", "p")))])[name]


CHT = Link("cht", (("solid", "right"), ("fluid", "wall")), NearestNeighborMapping())


def _pair(link: Link = CHT) -> dict[str, Context]:
    regions = contexts(["solid", "fluid"], [link], max_iterations=1)
    for ctx in regions.values():
        ctx.fields["T"] = np.array([1.0, 2.0])
    return regions


def _run(regions: dict[str, Context]) -> None:
    run({name: (ctx, [solve("T", lambda ctx: None)]) for name, ctx in regions.items()})


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
    assert received_flux.q.tolist() == [-1.0, -2.0]
    assert received_temperature.T.tolist() == [1.0, 2.0]


def test_each_region_gets_its_own_graph_with_its_coupling_op() -> None:
    regions = _pair()
    _Wall(regions["solid"], "right")
    _Flux(regions["fluid"], "wall")

    trees = {
        name: tree(graph(ctx, solve("T", lambda ctx: None)).ops) for name, ctx in regions.items()
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
        (s.region, s.patch, s.field, s.sends, s.receives, s.points)
        for s in (solid_side, fluid_side)
    ] == [
        ("solid", "right", "T", WallTemperature, WallHeatFlux, 2),
        ("fluid", "wall", "T", WallHeatFlux, WallTemperature, 2),
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
    assert (controller.label, controller.parallel) == ("fluid.coupling", False)
    assert [(end, s.field, s.partner) for end, s in controller.sides.items()] == [
        (("solid", "right"), "T", ("fluid", "wall")),
        (("fluid", "wall"), "T", ("solid", "right")),
    ]
    # what the sides send does not change: converged after one pass
    assert [(o.passes, o.converged) for o in controller.outcomes] == [(1, True)]


class _Other(BaseModel):
    flux: Array


@coupled
class _Odd(_Patch):
    """Sends a flux named otherwise, receives a heat flux, not a temperature."""

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
        "'q' is in WallHeatFlux (received by fluid.wall) only",
        "exchange 'solid-fluid': each side must receive what the other sends: "
        "'flux' is in _Other (sent by fluid.wall) only; "
        "'q' is in WallHeatFlux (received by solid.right) only",
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

    with pytest.raises(ValueError) as error:
        _run(regions)

    # the region that checked first raises: fluid receives, solid sends to the side without
    hint = "give its patch coordinates (@coupled.coordinates) or choose mapping 'identity'"
    assert str(error.value) in (
        "exchange 'cht' (solid.right -> fluid.wall): mapping 'nearest-neighbor' needs the "
        f"coordinates of both sides, but the receiving side has none: {hint}",
        "exchange 'cht' (fluid.wall -> solid.right): mapping 'nearest-neighbor' needs the "
        f"coordinates of both sides, but the sending side has none: {hint}",
    )


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
