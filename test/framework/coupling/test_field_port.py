# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Field ports: a region's field read or written on a location, through a fake field access.

The fake access keeps each field per location as a numpy array and logs the
writes; every patch has the faces of the fake geometry the test gives.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

import numpy as np
import pytest
from framework.coupling.regions import FakeGeometry, run, solve

from neofoam.framework.context import Context
from neofoam.framework.coupling import (
    CouplingLoop,
    Exchange,
    Global,
    IdentityMapping,
    InProcessCommChannel,
    Participant,
    Scalar,
    UniformMapping,
    ValueData,
    VolumeMeanMapping,
    coupled,
    coupling,
)
from neofoam.framework.coupling.exchanges import participant_exchanges
from neofoam.framework.coupling.field_port import FieldPort, check_field_ports


class _Fields:
    """A field access over arrays by ``(field, location name)``; logs every write."""

    def __init__(self, values: dict[tuple[str, str], Any]) -> None:
        self.values = {where: np.asarray(array, dtype=float) for where, array in values.items()}
        self.written: list[tuple[str, str, list[float]]] = []

    def read(self, field: str, location: Any) -> Any:
        return self.values[(field, location.name)]

    def write(self, field: str, location: Any, values: Any) -> None:
        self.values[(field, location.name)] = np.asarray(values, dtype=float)
        self.written.append((field, location.name, np.asarray(values).tolist()))


def _regions(
    exchange: Exchange,
    field_ports: dict[str, list[FieldPort]],
    fields: dict[str, Optional[_Fields]],
    geometry: Optional[dict[str, FakeGeometry]] = None,
    scheme: Literal["serial", "parallel"] = "serial",
) -> dict[str, Context]:
    regions = [region for region, _port in exchange.ends]
    loops = [CouplingLoop("coupling", (exchange.name,), max_iterations=1, scheme=scheme)]
    couplings = participant_exchanges(
        regions, [exchange], InProcessCommChannel(), loops, field_ports
    )
    return {
        region: Context(
            fields={},
            models={
                "coupling": Participant(
                    couplings[region], (geometry or {}).get(region, FakeGeometry()), fields[region]
                )
            },
            region=region,
        )
        for region in regions
    }


def _run(regions: dict[str, Context]) -> None:
    run({name: (ctx, [solve("T", lambda ctx: None)]) for name, ctx in regions.items()})


class _Mean(ValueData):
    T: Scalar


@coupled
class _Meter:
    def __init__(self) -> None:
        self.received: list[float] = []

    @coupled.receives
    def mean(self, data: _Mean) -> None:
        self.received.append(data.T)


def test_an_out_port_on_the_cells_sends_the_field_it_reads() -> None:
    """Cells -> Global: the volume mean of T = 300, 400 over the volumes 1, 3 is 375."""
    exchange = Exchange("mean", (("plant", "T_cells"), ("controller", "T")), VolumeMeanMapping())
    plant_fields = _Fields({("T", "T_cells"): [300.0, 400.0]})
    regions = _regions(
        exchange,
        {"plant": [FieldPort("T_cells", "T")]},
        {"plant": plant_fields, "controller": None},
        {"plant": FakeGeometry(cells=np.zeros((2, 1)), volumes=np.array([1.0, 3.0]))},
    )
    meter = _Meter()
    coupling(regions["controller"]).add(Global("T"), meter)

    _run(regions)

    assert meter.received == [375.0]
    assert plant_fields.written == []


class _Setpoint(ValueData):
    T: Scalar


@coupled
class _Controller:
    @coupled.sends
    def setpoint(self) -> _Setpoint:
        return _Setpoint(T=310.0)


def test_an_in_port_on_a_patch_writes_the_field_it_receives() -> None:
    """Global -> Patch: uniform puts the controller's 310 on both faces of the inlet."""
    exchange = Exchange("act", (("controller", "u"), ("plant", "T_in")), UniformMapping())
    plant_fields = _Fields({("T", "inlet"): [0.0, 0.0]})
    regions = _regions(
        exchange,
        {"plant": [FieldPort("T_in", "T", patch="inlet", direction="in")]},
        {"plant": plant_fields, "controller": None},
        {"plant": FakeGeometry(centres={"inlet": np.zeros((2, 1))})},
    )
    coupling(regions["controller"]).add(Global("u"), _Controller())

    _run(regions)

    assert plant_fields.written == [("T", "inlet", [310.0, 310.0])]


def test_inout_ports_exchange_their_fields_both_ways() -> None:
    """Parallel: each port receives what the other read before the pass."""
    exchange = Exchange("wall", (("a", "T_a"), ("b", "T_b")), IdentityMapping())
    fields = {
        "a": _Fields({("T", "right"): [1.0, 2.0]}),
        "b": _Fields({("T", "left"): [3.0, 4.0]}),
    }
    regions = _regions(
        exchange,
        {
            "a": [FieldPort("T_a", "T", patch="right", direction="inout")],
            "b": [FieldPort("T_b", "T", patch="left", direction="inout")],
        },
        dict(fields),
        scheme="parallel",
    )

    _run(regions)

    assert fields["a"].written == [("T", "right", [3.0, 4.0])]
    assert fields["b"].written == [("T", "left", [1.0, 2.0])]


def test_a_region_with_field_ports_needs_a_field_access() -> None:
    exchange = Exchange("act", (("controller", "u"), ("plant", "T_in")), UniformMapping())

    with pytest.raises(
        ValueError, match=r"region plant has the field ports \['T_in'\] .* gives no"
    ):
        _regions(
            exchange,
            {"plant": [FieldPort("T_in", "T", patch="inlet", direction="in")]},
            {"plant": None, "controller": None},
        )


_AB = [Exchange("x", (("a", "p"), ("b", "q")))]


@pytest.mark.parametrize(
    ("ports", "error"),
    [
        (
            {"a": [FieldPort("p", "T")], "b": [FieldPort("q", "T")]},
            "exchange 'x': p and q are both out, p is out, it cannot receive",
        ),
        (
            {
                "a": [FieldPort("p", "T", direction="in")],
                "b": [FieldPort("q", "T", direction="in")],
            },
            "exchange 'x': p and q are both in, p is in, it cannot send",
        ),
        (
            {"a": [FieldPort("p", "T", direction="inout")], "b": [FieldPort("q", "T")]},
            "exchange 'x': p is inout and q is out; inout pairs with inout only",
        ),
        (
            {
                "a": [
                    FieldPort("p", "T", patch="inlet", direction="in"),
                    FieldPort("r", "T", patch="inlet", direction="inout"),
                ]
            },
            "the field ports a.p and a.r both write T on patch inlet",
        ),
        ({"a": [FieldPort("p", "T"), FieldPort("r", "T")]}, r"field ports \['a.r'\] are in no ex"),
        ({"a": [FieldPort("p", "T"), FieldPort("p", "U")]}, "region a has two field ports named p"),
    ],
    ids=["out-out", "in-in", "inout-out", "written twice", "in no exchange", "named twice"],
)
def test_field_ports_are_checked_at_load(ports: dict[str, list[FieldPort]], error: str) -> None:
    with pytest.raises(ValueError, match=error):
        check_field_ports(_AB, ports)


@pytest.mark.parametrize("directions", [("out", "in"), ("in", "out"), ("inout", "inout")], ids=str)
def test_an_exchange_of_field_ports_connects_out_to_in_or_inout_to_inout(
    directions: tuple[str, str],
) -> None:
    first, second = directions
    ports = {
        "a": [FieldPort("p", "T", direction=first)],  # type: ignore[arg-type]
        "b": [FieldPort("q", "T", direction=second)],  # type: ignore[arg-type]
    }

    check_field_ports(_AB, ports)
