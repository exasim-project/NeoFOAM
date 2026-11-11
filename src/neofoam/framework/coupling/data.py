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

from dataclasses import dataclass
from typing import Annotated, Any, Callable, Optional

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


def _check_arrays(model: type[BaseModel]) -> None:
    """Raise ``TypeError`` unless every field of the exchange model ``model`` is :data:`Array`."""
    for name, info in model.model_fields.items():
        if not any(isinstance(meta, _ArrayKind) for meta in info.metadata):
            raise TypeError(
                f"the field '{name}' of the exchange model {model.__qualname__} is not Array: "
                "an exchange model holds Array fields only, one value per point of the patch"
            )


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
        # that both are pydantic models @coupled checked on the class
        _check_arrays(self.sends)
        _check_arrays(self.receives)
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
    """Raise ``ValueError`` unless ``side`` receives the fields its partner sends."""
    sent, received = announced["fields"], tuple(side.receives.model_fields)
    what_sent = f"{announced['model']} (sent by {announced['label']})"
    what_received = f"{side.receives.__qualname__} (received by {side.label})"
    problems = [f"'{n}' is in {what_sent} only" for n in sent if n not in received]
    problems += [f"'{n}' is in {what_received} only" for n in received if n not in sent]
    if problems:
        raise ValueError(
            f"exchange '{name}': each side must receive what the other sends: "
            + "; ".join(problems)
        )


def _sent(side: ExchangeSide, node: str, iteration: int, time: Any) -> Data:
    """What ``side`` sends now, as float copies.

    Raises ``FloatingPointError`` when an array of it is not finite (in coupling
    node ``node``, after pass ``iteration``, at ``time``).
    """
    sent = side.send()
    if not isinstance(sent, side.sends):
        raise TypeError(
            f"the coupled side {side.label} declares it sends {side.sends.__name__}, "
            f"but its send() returned {type(sent).__name__}"
        )
    data = {name: np.array(getattr(sent, name), dtype=float) for name in side.sends.model_fields}
    for name, values in data.items():
        if not np.all(np.isfinite(values)):
            when = f"after pass {iteration}" if iteration else "before the first pass"
            at = "" if time is None else f" (t={time})"
            raise FloatingPointError(
                f"coupling '{node}': the {name} of region {side.region} patch {side.patch} "
                f"is not finite {when}{at}"
            )
    return data
