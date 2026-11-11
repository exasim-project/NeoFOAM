# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""What a port exchanges: data models of :data:`Array` or :data:`Scalar` fields, their streams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, Callable, ClassVar, Optional

import numpy as np
from pydantic import BaseModel, PlainValidator

from .comm.channel import Data

if TYPE_CHECKING:
    from .coupled import CoupledPort

# (region, port)
End = tuple[str, str]


@dataclass(frozen=True)
class _ArrayKind:
    """Marks a field of a data model as :data:`Array`."""


@dataclass(frozen=True)
class _ScalarKind:
    """Marks a field of a data model as :data:`Scalar`."""


def _as_array(value: Any) -> Any:
    return np.asarray(value, dtype=float)


# A field of a data model: one value (or row) per entity of the location, in mesh order.
Array = Annotated[np.ndarray, _ArrayKind(), PlainValidator(_as_array)]

# A field of a ValueData model: one number.
Scalar = Annotated[float, _ScalarKind(), PlainValidator(float)]


class _Data(BaseModel):
    """The data models' base: ``entity`` names what one row is."""

    entity: ClassVar[str] = ""
    field_kind: ClassVar[type] = _ArrayKind


class FaceData(_Data):
    """Data with one row per face of a :class:`~neofoam.framework.coupling.Patch`.

    Example::

        class WallTemperature(FaceData):
            T: Array
    """

    entity: ClassVar[str] = "face"


class CellData(_Data):
    """Data with one row per cell of :class:`~neofoam.framework.coupling.Cells`."""

    entity: ClassVar[str] = "cell"


class ValueData(_Data):
    """One value per field (:data:`Scalar`), on a :class:`~neofoam.framework.coupling.Global`."""

    entity: ClassVar[str] = "value"
    field_kind: ClassVar[type] = _ScalarKind


class ProbeData(_Data):
    """Data with one row per point of a ``probe`` mapping, on a ``Global``."""

    entity: ClassVar[str] = "probe"


def base_name(model: type[_Data]) -> str:
    """The base of ``model``: ``FaceData``, ``CellData``, ``ValueData`` or ``ProbeData``."""
    return next(
        base.__name__
        for base in (FaceData, CellData, ValueData, ProbeData)
        if issubclass(model, base)
    )


def check_model(model: Any, owner: str) -> type[_Data]:
    """``model`` if it is a data model with fields of its kind; ``TypeError`` naming ``owner``."""
    if not (isinstance(model, type) and issubclass(model, _Data) and model.entity):
        raise TypeError(
            f"{owner} declares {model!r}, which is no FaceData, CellData, ValueData or ProbeData"
        )
    kind = model.field_kind
    for name, info in model.model_fields.items():
        if not any(isinstance(meta, kind) for meta in info.metadata):
            wanted = "Scalar" if kind is _ScalarKind else "Array"
            raise TypeError(
                f"{model.__qualname__}.{name} is not {wanted}: a {base_name(model)} holds "
                f"{wanted} fields only"
            )
    if not model.model_fields:
        raise TypeError(f"{model.__qualname__} has no field")
    return model


def key(model: type[BaseModel]) -> str:
    """What pairs a sent with a received model: their field names, ``T`` or ``T_sink,h``."""
    return ",".join(sorted(model.model_fields))


@dataclass(frozen=True)
class Stream:
    """One data model a port sends or receives, and the bound method that does it."""

    model: type[Any]
    method: Callable[..., Any]

    @property
    def key(self) -> str:
        return key(self.model)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.model.model_fields)


def check_received(name: str, announced: dict[str, Any], port: CoupledPort) -> None:
    """Raise ``ValueError`` unless ``port`` receives exactly what its partner sends."""
    sent: dict[str, str] = announced["sends"]
    received = {stream.key: stream.model.__qualname__ for stream in port.receives}
    problems = [
        f"{announced['label']} sends {model}({fields}), {port.label} receives nothing with {fields}"
        for fields, model in sent.items()
        if fields not in received
    ]
    problems += [
        f"{port.label} receives {model}({fields}), {announced['label']} sends nothing with {fields}"
        for fields, model in received.items()
        if fields not in sent
    ]
    if problems:
        raise ValueError(f"exchange '{name}': " + "; ".join(problems))


def check_rows(stream: Stream, data: Data, size: Optional[int], port: CoupledPort) -> None:
    """Raise ``ValueError`` unless ``data`` has a row per entity of the port (``size``)."""
    entity = stream.model.entity
    if size is None or entity not in ("face", "cell"):
        return
    for name, values in data.items():
        if len(values) != size:
            raise ValueError(
                f"{port.label} sends {len(values)} rows of {stream.model.__qualname__}.{name}, "
                f"but its location {port.location} has {size} {entity}s; an array has one row "
                "per entity, in mesh order"
            )


def sent(stream: Stream, port: CoupledPort, loop: str, when: tuple[int, Any]) -> Data:
    """What ``stream`` sends now, as float copies; ``FloatingPointError`` when not finite.

    ``when``: the pass (from 0) and the time.
    """
    model = stream.method()
    if not isinstance(model, stream.model):
        raise TypeError(
            f"{port.label} declares it sends {stream.model.__name__}, but its "
            f"@coupled.sends returned {type(model).__name__}"
        )
    data = {name: np.array(getattr(model, name), dtype=float) for name in stream.names}
    for name, values in data.items():
        if not np.all(np.isfinite(values)):
            iteration, time = when
            after = f"after pass {iteration}" if iteration else "before the first pass"
            at = "" if time is None else f" (t={time})"
            raise FloatingPointError(
                f"coupling '{loop}': the {name} of {port.label} is not finite {after}{at}"
            )
    return data
