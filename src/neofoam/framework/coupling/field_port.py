# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Field ports: a region's field read (``out``) or written (``in``) on a location, from the yaml.

A field port exists once, for every solver: the case names it in the ``ports:``
section of ``system/regionCoupling.yaml`` and the region's :class:`Participant`
binds it through the :class:`FieldAccess` its solver gives. Writing changes the
solution, so ``in`` and ``inout`` are explicit; ``out`` only reads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Mapping, Optional, Protocol, Sequence

from pydantic import create_model

from .coupled import coupled
from .data import Array, CellData, FaceData
from .location import Cells, Location, Patch

if TYPE_CHECKING:
    from .exchanges import Exchange

Direction = Literal["out", "in", "inout"]


class FieldAccess(Protocol):
    """Reads and writes a region's field on a location, in mesh order; one per backend."""

    def read(self, field: str, location: Location) -> Any:
        """The values of ``field`` on ``location``: a row per face or cell."""

    def write(self, field: str, location: Location, values: Any) -> None:
        """Set ``field`` on ``location`` to ``values``, a row per face or cell."""


@dataclass(frozen=True)
class FieldPort:
    """A field port of a region: ``field`` on ``patch`` (``None``: on the cells), its direction.

    Example::

        FieldPort("T_out", "T", patch="outlet")              # out: a sensor
        FieldPort("U_in", "U", patch="inlet", direction="in")  # an actuator
    """

    name: str
    field: str
    patch: Optional[str] = None
    direction: Direction = "out"

    @property
    def location(self) -> Location:
        """``Patch(patch)``, or the cells under the port's name."""
        return Patch(self.patch) if self.patch is not None else Cells(self.name)

    @property
    def sends(self) -> bool:
        return self.direction in ("out", "inout")

    @property
    def receives(self) -> bool:
        return self.direction in ("in", "inout")


def instance(port: FieldPort, access: FieldAccess) -> Any:
    """The ``@coupled`` instance of ``port``: it reads and writes through ``access``."""
    base = FaceData if port.patch is not None else CellData
    model: type[Any] = create_model(  # type: ignore[call-overload]
        f"{base.__name__[:-4]}{port.field}", __base__=base, **{port.field: (Array, ...)}
    )
    location = port.location

    def send(self: Any) -> Any:
        return model.model_validate({port.field: access.read(port.field, location)})

    def receive(self: Any, data: Any) -> None:
        access.write(port.field, location, getattr(data, port.field))

    send.__annotations__ = {"return": model}
    receive.__annotations__ = {"data": model, "return": None}
    methods: dict[str, Any] = {}
    if port.sends:
        methods["send"] = coupled.sends(send)
    if port.receives:
        methods["receive"] = coupled.receives(receive)
    return coupled(type(f"FieldPort[{port.name}]", (), methods))()


def check_field_ports(
    exchanges: Sequence[Exchange], field_ports: Mapping[str, Sequence[FieldPort]]
) -> None:
    """Raise ``ValueError`` unless every field port is in an exchange that fits its direction.

    An exchange of two field ports connects ``out`` to ``in`` or ``inout`` to
    ``inout``; a field on a location is written by one port.
    """
    by_end = {(region, port.name): port for region, ports in field_ports.items() for port in ports}
    _check_names(field_ports)
    _check_written_once(field_ports)
    ends = {end for exchange in exchanges for end in exchange.ends}
    lonely = sorted(f"{region}.{name}" for region, name in by_end if (region, name) not in ends)
    if lonely:
        raise ValueError(f"the field ports {lonely} are in no exchange")
    for exchange in exchanges:
        first, second = (by_end.get(end) for end in exchange.ends)
        if first is not None and second is not None:
            _check_directions(exchange, first, second)


def _check_names(field_ports: Mapping[str, Sequence[FieldPort]]) -> None:
    for region, ports in field_ports.items():
        names = [port.name for port in ports]
        twice = sorted({name for name in names if names.count(name) > 1})
        if twice:
            raise ValueError(f"region {region} has two field ports named {twice[0]}")


def _check_written_once(field_ports: Mapping[str, Sequence[FieldPort]]) -> None:
    for region, ports in field_ports.items():
        writer: dict[tuple[str, Optional[str]], str] = {}
        for port in ports:
            if not port.receives:
                continue
            where = (port.field, port.patch)
            if where in writer:
                on = f"patch {port.patch}" if port.patch is not None else "the cells"
                raise ValueError(
                    f"the field ports {region}.{writer[where]} and {region}.{port.name} both "
                    f"write {port.field} on {on}; one port writes a field on a location"
                )
            writer[where] = port.name


def _check_directions(exchange: Exchange, first: FieldPort, second: FieldPort) -> None:
    pair = {first.direction, second.direction}
    if pair in ({"out", "in"}, {"inout"}):
        return
    if "inout" in pair:
        raise ValueError(
            f"exchange '{exchange.name}': {first.name} is {first.direction} and {second.name} "
            f"is {second.direction}; inout pairs with inout only"
        )
    cannot = "send" if first.direction == "in" else "receive"
    raise ValueError(
        f"exchange '{exchange.name}': {first.name} and {second.name} are both "
        f"{first.direction}, {first.name} is {first.direction}, it cannot {cannot}; an "
        "exchange connects out to in"
    )
