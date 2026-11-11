# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""What a coupled side exchanges: :data:`Array` fields, the side, and what it sends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, Callable, Optional

import numpy as np
from pydantic import BaseModel, PlainValidator

from .comm_channel import Data

# (region, patch)
End = tuple[str, str]


@dataclass(frozen=True)
class _ArrayKind:
    """Marks a field of an exchange model as :data:`Array`."""


def _as_array(value: Any) -> Any:
    return np.asarray(value, dtype=float)


# A field of an exchange model: one value (or row) per point of the patch.
Array = Annotated[np.ndarray, _ArrayKind(), PlainValidator(_as_array)]


def _is_model(model: Any) -> bool:
    return isinstance(model, type) and issubclass(model, BaseModel)


def _check_arrays(model: type[BaseModel]) -> None:
    for name, info in model.model_fields.items():
        if not any(isinstance(meta, _ArrayKind) for meta in info.metadata):
            raise TypeError(
                f"the field '{name}' of the exchange model {model.__qualname__} is not Array: "
                "an exchange model holds Array fields only, one value per point of the patch"
            )


@dataclass(frozen=True)
class ExchangeSide:
    """A coupled side of a region: its patch, the field it couples, its models and points.

    ``receive`` gets data already mapped onto its ``coordinates`` (one row per
    point, or ``None``). Built by
    :meth:`~neofoam.framework.coupling.Coupling.add` for a ``@coupled`` instance.
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
        _check_arrays(self.sends)
        _check_arrays(self.receives)
        if self.coordinates is not None:
            object.__setattr__(self, "coordinates", np.asarray(self.coordinates, dtype=float))

    @property
    def end(self) -> End:
        """The side's end of its exchange."""
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


def _check_received(name: str, announced: dict[str, Any], side: ExchangeSide) -> None:
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
    """What ``side`` sends now, as float copies; ``FloatingPointError`` when not finite."""
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
