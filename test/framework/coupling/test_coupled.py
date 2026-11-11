# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""``@coupled`` classes, their binding to locations and the exchanges that join them.

``_Wall`` sends its wall temperature and receives a heat flux, ``_Flux`` the
reverse: one model per direction, each region on its own thread. The module
uses ``from __future__ import annotations`` on purpose: ``@coupled`` resolves
the string annotations. Every patch has the two faces at x = 0 and 1 of the
fake geometry unless a test says otherwise.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from framework.coupling.regions import contexts, couple, graph, run, solve, tree

from neofoam.framework.context import Context
from neofoam.framework.coupling import (
    Array,
    Cells,
    Exchange,
    FaceData,
    Global,
    IdentityMapping,
    NearestNeighborMapping,
    Patch,
    Scalar,
    ValueData,
    coupled,
    coupling,
)
from neofoam.framework.coupling.loop import CouplingOp
from neofoam.framework.types import Placement


class WallTemperature(FaceData):
    T: Array


class WallHeatFlux(FaceData):
    q: Array


class _Patch:
    def __init__(self, ctx: Context) -> None:
        self.ctx = ctx
        self.received: list[Any] = []


@coupled
class _Wall(_Patch):
    """Sends its temperature, receives a flux."""

    @coupled.sends
    def wall_temperature(self) -> WallTemperature:
        return WallTemperature(T=self.ctx.fields["T"])

    @coupled.receives
    def heat_in(self, data: WallHeatFlux) -> None:
        self.received.append(data)


@coupled
class _Flux(_Patch):
    """Sends a flux, receives a temperature."""

    @coupled.sends
    def flux(self) -> WallHeatFlux:
        return WallHeatFlux(q=-self.ctx.fields["T"])

    @coupled.receives
    def temperature(self, data: WallTemperature) -> None:
        self.received.append(data)


def _bind(ctx: Context, patch: str, cls: type[_Patch], points: Any = ((0.0,), (1.0,))) -> Any:
    """``cls`` on ``patch`` of ``ctx``, its faces at ``points``, bound as a solver binds it."""
    instance = cls(ctx)
    coupling(ctx).geometry.centres[patch] = None if points is None else np.array(points)  # type: ignore[attr-defined]
    coupling(ctx).add(Patch(patch), instance)
    return instance


def _region(name: str) -> Context:
    return contexts([name, "other"], [Exchange("x", ((name, "right"), ("other", "p")))])[name]


CHT = Exchange("cht", (("solid", "right"), ("fluid", "wall")), NearestNeighborMapping())


def _pair(exchange: Exchange = CHT, **node: Any) -> dict[str, Context]:
    regions = contexts(["solid", "fluid"], [exchange], **{"max_iterations": 1, **node})
    for ctx in regions.values():
        ctx.fields["T"] = np.array([1.0, 2.0])
    return regions


def _run(regions: dict[str, Context]) -> None:
    run({name: (ctx, [solve("T", lambda ctx: None)]) for name, ctx in regions.items()})


def test_coupled_reads_the_models_off_the_string_annotations_of_any_methods() -> None:
    declaration = _Wall.__coupled__  # type: ignore[attr-defined]

    assert (declaration.sends, declaration.receives) == (
        (("wall_temperature", WallTemperature),),
        (("heat_in", WallHeatFlux),),
    )


def test_coupled_needs_a_sends_or_receives_method() -> None:
    with pytest.raises(TypeError, match="_Unmarked is @coupled but marks no @coupled.sends or"):

        @coupled
        class _Unmarked(_Patch):
            def send(self) -> WallTemperature:
                raise NotImplementedError


def test_coupled_needs_a_data_model_as_annotation() -> None:
    with pytest.raises(
        TypeError,
        match=r"_NoModel.send \(@coupled.sends\) declares <class 'float'>, which is no FaceData",
    ):

        @coupled
        class _NoModel(_Patch):
            @coupled.sends
            def send(self) -> float:
                raise NotImplementedError


class _Plain(FaceData):
    q: Array
    h: float


class _Vector(ValueData):
    U: Array


@pytest.mark.parametrize(
    ("model", "error"),
    [
        (_Plain, "_Plain.h is not Array: a FaceData holds Array fields only"),
        (_Vector, "_Vector.U is not Scalar: a ValueData holds Scalar fields only"),
    ],
    ids=["FaceData", "ValueData"],
)
def test_coupled_checks_the_fields_of_a_data_model(model: type, error: str) -> None:
    def send(self: Any) -> Any:
        raise NotImplementedError

    send.__annotations__ = {"return": model}

    with pytest.raises(TypeError, match=error):
        coupled(type("_Sender", (), {"send": coupled.sends(send)}))


class _Temperatures(FaceData):
    T: Array
    h: Array


def test_coupled_sends_a_data_field_at_most_once() -> None:
    with pytest.raises(
        TypeError, match=r"two @coupled.sends of T in _Twice \(WallTemperature and _Temperatures\)"
    ):

        @coupled
        class _Twice(_Patch):
            @coupled.sends
            def first(self) -> WallTemperature:
                raise NotImplementedError

            @coupled.sends
            def second(self) -> _Temperatures:
                raise NotImplementedError


def test_only_a_coupled_instance_is_added() -> None:
    with pytest.raises(TypeError, match="_Patch is not @coupled"):
        coupling(_region("a")).add(Patch("right"), _Patch(_region("a")))


def test_a_data_model_lives_on_its_location_only() -> None:
    region = _region("a")

    with pytest.raises(TypeError, match=r"WallTemperature is FaceData, which does not live on "):
        coupling(region).add(Cells("right"), _Wall(region))


def test_the_owner_binds_a_coupled_instance_to_its_location() -> None:
    region = _region("solid")

    wall = _bind(region, "right", _Wall)

    (operation,) = coupling(region).operations()
    assert isinstance(operation.func, CouplingOp)
    assert (operation.operation_name, operation.metadata.wraps) == ("solid.coupling", Placement())
    (port,) = operation.func.ports
    assert (port.region, port.name, port.location) == ("solid", "right", Patch("right"))
    assert [(s.model, s.method) for s in port.sends] == [(WallTemperature, wall.wall_temperature)]
    assert [(s.model, s.method) for s in port.receives] == [(WallHeatFlux, wall.heat_in)]


@coupled
class _Sensor(_Patch):
    @coupled.sends
    def temperature(self) -> WallTemperature:
        return WallTemperature(T=self.ctx.fields["T"])


@coupled
class _Actuator(_Patch):
    @coupled.receives
    def heat_in(self, data: WallHeatFlux) -> None:
        self.received.append(data)


def test_instances_on_one_location_merge_into_its_port() -> None:
    regions = _pair()
    sensor = _bind(regions["solid"], "right", _Sensor)
    actuator = _bind(regions["solid"], "right", _Actuator)
    flux = _bind(regions["fluid"], "wall", _Flux)

    _run(regions)

    assert [data.q.tolist() for data in actuator.received] == [[-1.0, -2.0]]
    assert [data.T.tolist() for data in flux.received] == [[1.0, 2.0]]
    assert sensor.received == []


def test_two_instances_on_one_port_send_a_data_field_once() -> None:
    region = _region("solid")
    _bind(region, "right", _Sensor)

    with pytest.raises(ValueError, match=r"two @coupled.sends of T on solid.right"):
        _bind(region, "right", _Wall)


def test_coupled_exchange_with_one_model_per_direction() -> None:
    regions = _pair()
    wall, flux = _bind(regions["solid"], "right", _Wall), _bind(regions["fluid"], "wall", _Flux)

    _run(regions)

    (received_flux,) = wall.received
    (received_temperature,) = flux.received
    assert isinstance(received_flux, WallHeatFlux)
    assert isinstance(received_temperature, WallTemperature)
    assert received_flux.q.tolist() == [-1.0, -2.0]
    assert received_temperature.T.tolist() == [1.0, 2.0]


def test_each_region_gets_its_own_graph_with_its_coupling_op() -> None:
    regions = _pair()
    _bind(regions["solid"], "right", _Wall)
    _bind(regions["fluid"], "wall", _Flux)

    trees = {
        name: tree(graph(ctx, solve("T", lambda ctx: None)).ops) for name, ctx in regions.items()
    }

    assert trees == {
        "solid": [("solid.coupling", ["solve_T"])],
        "fluid": [("fluid.coupling", ["solve_T"])],
    }


def test_coupling_op_public_read_api() -> None:
    regions = _pair()
    _bind(regions["solid"], "right", _Wall)
    _bind(regions["fluid"], "wall", _Flux)
    solid_op, fluid_op = (couple(ctx) for ctx in regions.values())

    (solid_port,), (fluid_port,) = solid_op.ports, fluid_op.ports
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
    assert (solid_port.label, fluid_port.label) == ("solid.right", "fluid.wall")
    assert (solid_op.partner(solid_port), fluid_op.partner(fluid_port)) == (
        "fluid.wall",
        "solid.right",
    )
    assert solid_op.exchange_of(solid_port) is CHT
    # the leader builds its controller at its first loop test
    assert (solid_op.controller, fluid_op.controller) == (None, None)

    _run(regions)

    controller = fluid_op.controller
    assert solid_op.controller is None and controller is not None
    assert (controller.label, controller.parallel) == ("fluid.coupling", False)
    assert [(stream_id, s.partner) for stream_id, s in controller.streams.items()] == [
        (("solid", "right", "T"), ("fluid", "wall")),
        (("fluid", "wall", "q"), ("solid", "right")),
    ]
    assert [(o.passes, o.converged) for o in controller.outcomes] == [(1, True)]


class _Other(FaceData):
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


def test_an_exchange_pairs_what_one_port_sends_with_what_the_other_receives() -> None:
    regions = _pair(Exchange("solid-fluid", (("solid", "right"), ("fluid", "wall"))))
    _bind(regions["solid"], "right", _Wall)
    _bind(regions["fluid"], "wall", _Odd)

    with pytest.raises(ValueError) as error:
        _run(regions)

    # the region that checked first raises; the other learns that it left
    assert str(error.value) in (
        "exchange 'solid-fluid': solid.right sends WallTemperature(T), fluid.wall receives "
        "nothing with T; fluid.wall receives WallHeatFlux(q), solid.right sends nothing with q",
        "exchange 'solid-fluid': fluid.wall sends _Other(flux), solid.right receives nothing "
        "with flux; solid.right receives WallHeatFlux(q), fluid.wall sends nothing with q",
    )


def test_exchange_with_nearest_neighbor_needs_the_points_of_both_ports() -> None:
    regions = _pair()
    _bind(regions["solid"], "right", _Wall)
    _bind(regions["fluid"], "wall", _Flux, points=None)

    with pytest.raises(ValueError) as error:
        _run(regions)

    # the region that checked first raises: either direction lacks the points
    hint = "give its mesh geometry or choose mapping 'identity'"
    assert str(error.value) in (
        "exchange 'cht' (solid.right -> fluid.wall): mapping 'nearest-neighbor' needs the "
        f"points of both ports, but the receiving port has none: {hint}",
        "exchange 'cht' (fluid.wall -> solid.right): mapping 'nearest-neighbor' needs the "
        f"points of both ports, but the sending port has none: {hint}",
    )


def test_a_port_sends_a_row_per_entity_of_its_location() -> None:
    regions = _pair()
    _bind(regions["solid"], "right", _Wall, points=((0.0,), (0.5,), (1.0,)))
    _bind(regions["fluid"], "wall", _Flux)

    with pytest.raises(ValueError, match=r"solid.right sends 2 rows of WallTemperature.T, but"):
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
    exchanges = [
        Exchange("a-b", (("a", "to_b"), ("b", "to_a"))),
        Exchange("b-c", (("b", "to_c"), ("c", "to_b"))),
    ]
    regions = contexts(["a", "b", "c"], exchanges, max_iterations=1)
    for ctx in regions.values():
        ctx.fields["T"] = np.array([1.0, 2.0])
    _bind(regions["a"], "to_b", _Same)
    _bind(regions["b"], "to_a", _Same)
    _bind(regions["b"], "to_c", _Same)
    _bind(regions["c"], "to_b", _Same)
    ops = {name: couple(ctx) for name, ctx in regions.items()}

    _run(regions)

    assert [[port.name for port in op.ports] for op in ops.values()] == [
        ["to_b"],
        ["to_a", "to_c"],
        ["to_b"],
    ]
    assert [op.is_leader for op in ops.values()] == [False, False, True]
    controller = ops["c"].controller
    assert controller is not None
    # serial: a stream is received fresh when its sender steps before its receiver
    assert {stream: controller.fresh(stream) for stream in controller.streams} == {
        ("a", "to_b", "T"): True,
        ("b", "to_a", "T"): False,
        ("b", "to_c", "T"): True,
        ("c", "to_b", "T"): False,
    }


class _MeanTemperature(ValueData):
    T: Scalar


@coupled
class _MeanMeter(_Patch):
    @coupled.receives
    def mean(self, data: _MeanTemperature) -> None:
        self.received.append(data)


def test_a_transfer_mapping_maps_no_faces_onto_a_value() -> None:
    regions = _pair()
    _bind(regions["solid"], "right", _Sensor)
    coupling(regions["fluid"]).add(Global("wall"), _MeanMeter(regions["fluid"]))

    with pytest.raises(
        ValueError, match="'nearest-neighbor' maps face -> face, cell -> cell, not face -> value"
    ):
        _run(regions)


def test_identity_maps_the_data_of_ports_without_points() -> None:
    regions = _pair(Exchange("cht", (("solid", "right"), ("fluid", "wall")), IdentityMapping()))
    wall = _bind(regions["solid"], "right", _Wall, points=None)
    _bind(regions["fluid"], "wall", _Flux, points=None)

    _run(regions)

    assert [data.q.tolist() for data in wall.received] == [[-1.0, -2.0]]
