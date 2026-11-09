# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""What a coupled side exchanges: :data:`Array`, the side, and what it sends.

- :data:`Array` marks a field of an exchange model with one value (or row) per
  point of the patch; the mapping maps it onto the receiving side's points. The
  other fields of an exchange model are plain data, one value for the interface;
- an **exchange side** (:class:`ExchangeSide`) is what a boundary condition
  declares on its patch, as its region's coupling model added it: its region,
  patch and field, what it sends and receives (one pydantic model each) and its
  points; an **end** (:data:`End`) is its ``(region, patch)``;
- an **exchange** (:class:`Exchange`) joins a side with the side of another
  region (its partner); the case configures it per patch pair; its **mapping**
  (:class:`~neofoam.framework.coupling.kernels.CouplingMapping`) maps the :data:`Array`
  data the partner sends onto the side's points;
- :func:`_sent` takes what a side sends now as plain data and checks that its
  :data:`Array` data is finite; :func:`_check_received` that a side receives
  what its partner sends.
"""

from __future__ import annotations

import types
import typing
from dataclasses import dataclass
from typing import Annotated, Any, Callable, Optional, Sequence

import numpy as np
from pydantic import BaseModel, PlainValidator

from .kernels import CouplingMapping
from .transport import Data

# The end of an exchange: (region, patch).
End = tuple[str, str]


@dataclass(frozen=True)
class _ArrayKind:
    """Marks a field of an exchange model as :data:`Array`."""


def _as_array(value: Any) -> Any:
    return np.asarray(value, dtype=float)


# A field of an exchange model with one value (or row) per point of the patch; the
# mapping maps it onto the receiving side's points.
Array = Annotated[np.ndarray, _ArrayKind(), PlainValidator(_as_array)]


def _is_model(model: Any) -> bool:
    return isinstance(model, type) and issubclass(model, BaseModel)


def _kinds(model: type[BaseModel]) -> dict[str, str]:
    """The kind of each field of the exchange model ``model``: ``Array``, or its type's name."""
    return {
        name: "Array"
        if any(isinstance(meta, _ArrayKind) for meta in info.metadata)
        else getattr(info.annotation, "__name__", repr(info.annotation))
        for name, info in model.model_fields.items()
    }


# The types a plain (non-Array) field of an exchange model may have, besides a
# pydantic model of such fields.
_PLAIN = (float, int, str, bool, type(None))


def _plain(annotation: Any) -> bool:
    """Whether a field annotated ``annotation`` holds plain data: a :data:`_PLAIN` type,
    a union of them, or a pydantic model of such fields."""
    if annotation in _PLAIN:
        return True
    if _is_model(annotation):
        return all(_plain(info.annotation) for info in annotation.model_fields.values())
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        return all(_plain(arg) for arg in typing.get_args(annotation))
    return False


def _check_plain(model: type[BaseModel]) -> None:
    """Raise ``TypeError`` unless each plain (non-Array) field of ``model`` holds plain data."""
    for name, kind in _kinds(model).items():
        annotation = model.model_fields[name].annotation
        if kind != "Array" and not _plain(annotation):
            raise TypeError(
                f"the field '{name}' of the exchange model {model.__qualname__} is "
                f"{annotation!r}: a field that is not Array is float, int, str, bool or None, "
                "or a pydantic model of such fields"
            )


def _array_names(model: type[BaseModel]) -> tuple[str, ...]:
    """The :data:`Array` fields of the exchange model ``model``."""
    return tuple(name for name, kind in _kinds(model).items() if kind == "Array")


def _arrays(names: Sequence[str], data: Data) -> Data:
    """The :data:`Array` fields ``names`` of ``data``."""
    return {name: data[name] for name in names}


@dataclass(frozen=True)
class ExchangeSide:
    """A coupled side of a region: where it is, what it exchanges and where its points are.

    ``region`` and ``patch`` are its end of the exchange, ``field`` the field
    it couples, whose solution its region's coupling op repeats. ``sends`` and
    ``receives`` are pydantic models: :data:`Array` fields (one value per point
    of the patch, mapped onto the partner's points) and plain fields (one value
    for the interface, handed over as they are). The same model both ways is
    the special case of a symmetric exchange. ``send()`` returns an instance of
    ``sends``; ``receive(data)`` takes an instance of ``receives``, already
    mapped onto this side's points. ``coordinates`` holds one row per point of
    the patch, or ``None``; it is kept as a float numpy array.

    What travels between regions is plain data (it pickles): the plain fields
    are ``float``, ``int``, ``str``, ``bool`` or ``None`` (or a union of them),
    or a pydantic model of such fields, sent as its ``model_dump()``. Raises
    ``TypeError`` for another field type.

    A boundary condition declares it with
    :func:`~neofoam.framework.coupling.coupled`;
    :meth:`~neofoam.framework.coupling.Coupling.add` builds it (read-only).

    Example::

        ExchangeSide("solid", "right", "T", sends=WallTemperature, receives=WallHeatFlux,
                     send=self.send, receive=self.receive, coordinates=points)
    """

    region: str
    patch: str
    field: str
    sends: type[BaseModel]
    receives: type[BaseModel]
    send: Callable[[], BaseModel]
    receive: Callable[[Any], None]
    coordinates: Any = None

    def __post_init__(self) -> None:
        for role in ("sends", "receives"):
            if not _is_model(getattr(self, role)):
                raise TypeError(
                    f"a coupled side declares what it {role} as a pydantic model, "
                    f"got {getattr(self, role)!r}"
                )
            _check_plain(getattr(self, role))
        if self.coordinates is not None:
            object.__setattr__(self, "coordinates", np.asarray(self.coordinates, dtype=float))

    @property
    def end(self) -> End:
        """``(region, patch)``: the side's end of its exchange."""
        return (self.region, self.patch)

    @property
    def points(self) -> Optional[int]:
        """The number of points of the patch (rows of ``coordinates``), or ``None``."""
        if self.coordinates is None:
            return None
        return int(np.shape(self.coordinates)[0])

    @property
    def mapped(self) -> tuple[str, ...]:
        """The fields of what the side receives that the mapping maps (its :data:`Array` fields).

        The others are one value for the interface, handed over as they are.
        """
        return _array_names(self.receives)

    @property
    def label(self) -> str:
        """``<region>.<patch>``."""
        return f"{self.region}.{self.patch}"


@dataclass(frozen=True)
class Exchange:
    """The exchange of a side, as its region sees it: its name, the partner and the mapping.

    The case configures it per patch pair (``system/regionCoupling.yaml``,
    ``exchanges:``). The side receives what the partner side, on patch
    ``partner[1]`` of region ``partner[0]``, sends, mapped onto its points by
    ``mapping``.
    """

    name: str
    partner: End
    mapping: CouplingMapping


def _check_received(name: str, announced: dict[str, Any], side: ExchangeSide) -> None:
    """Raise ``ValueError`` unless ``side`` receives what its partner (``announced``) sends.

    By field name and kind (``Array`` or plain type); the model classes may differ.
    """
    sent, received = announced["kinds"], _kinds(side.receives)
    what_sent = f"{announced['model']} (sent by {announced['label']})"
    what_received = f"{side.receives.__qualname__} (received by {side.label})"
    problems = [f"'{n}' is in {what_sent} only" for n in sent if n not in received]
    problems += [f"'{n}' is in {what_received} only" for n in received if n not in sent]
    problems += [
        f"'{n}' is {kind} in {what_sent} but {received[n]} in {what_received}"
        for n, kind in sent.items()
        if n in received and received[n] != kind
    ]
    if problems:
        raise ValueError(
            f"exchange '{name}': each side must receive what the other sends: "
            + "; ".join(problems)
        )


def _dumped(sent: BaseModel, name: str) -> Any:
    """The plain field ``name`` of ``sent``; a pydantic model as its ``model_dump()``."""
    value = getattr(sent, name)
    return value.model_dump() if isinstance(value, BaseModel) else value


def _sent(side: ExchangeSide, node: str, iteration: int, time: Any) -> Data:
    """What ``side`` sends now, as named copies.

    Raises ``FloatingPointError`` when an :data:`Array` of it is not finite (in
    coupling node ``node``, after pass ``iteration``, at ``time``).
    """
    sent = side.send()
    if not isinstance(sent, side.sends):
        raise TypeError(
            f"the coupled side {side.label} declares it sends {side.sends.__name__}, "
            f"but its send() returned {type(sent).__name__}"
        )
    data = {
        name: np.array(getattr(sent, name), dtype=float) if kind == "Array" else _dumped(sent, name)
        for name, kind in _kinds(side.sends).items()
    }
    for name in _array_names(side.sends):
        if not np.all(np.isfinite(data[name])):
            when = f"after pass {iteration}" if iteration else "before the first pass"
            at = "" if time is None else f" (t={time})"
            raise FloatingPointError(
                f"coupling '{node}': the {name} of region {side.region} patch {side.patch} "
                f"is not finite {when}{at}"
            )
    return data
