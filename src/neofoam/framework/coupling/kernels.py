# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Coupling strategies: mapping, acceleration and convergence of the exchanged data."""

from __future__ import annotations

from typing import Any, Callable, Literal, NamedTuple, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from neofoam.core.plugin_system import PluginSystem

from .transport import Data


class SideChange(NamedTuple):
    """What one side sent in the previous and in the current pass, for a convergence check."""

    field: str
    previous: Data
    current: Data


# elements of the target-by-source distance block computed at once
_BLOCK_ELEMENTS = 1 << 20


def nearest_neighbor(source: Any, target: Any) -> Any:
    """For every target point, the index of the nearest source point (the lowest on a tie).

    Points are rows of ``(n_points, dim)`` arrays; a block of target points at a
    time, so memory is O(n + m), time O(n * m).

    Example::

        values_on_target = values_on_source[nearest_neighbor(source, target)]
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    indices = np.empty(len(target), dtype=np.intp)
    rows = max(1, _BLOCK_ELEMENTS // max(1, source.size))
    for start in range(0, len(target), rows):
        block = target[start : start + rows]
        distances = np.linalg.norm(block[:, None, :] - source[None, :, :], axis=-1)
        indices[start : start + rows] = np.argmin(distances, axis=1)
    return indices


# a mapping bound to both sides' points: data on the source points -> on the target's
BoundMapping = Callable[[Data], Data]


@PluginSystem.register(discriminator_variable="mapping", discriminator="type")
class CouplingMapping(BaseModel):
    """Maps the data a side sends onto the other side's points; a plugin family.

    Bound once, when the receiving op connects, to the partner's points (``source``)
    and its own (``target``; ``None`` without coordinates). A mapping of one's own
    subclasses it with a ``type: Literal[...]``, overrides :meth:`bind` and
    registers (``@CouplingMapping.register``); so do accelerations and convergences.
    """

    model_config = ConfigDict(extra="forbid")

    def bind(self, source: Any, target: Any) -> BoundMapping:
        """The mapping from ``source`` onto ``target``; ``ValueError`` when it cannot map."""
        raise NotImplementedError


@CouplingMapping.register
class IdentityMapping(CouplingMapping):
    """Hands the data over unchanged: the sides share their points."""

    type: Literal["identity"] = "identity"

    def bind(self, source: Any, target: Any) -> BoundMapping:
        if source is not None and target is not None and len(source) != len(target):
            raise ValueError(
                f"mapping '{self.type}' needs as many points on both sides: "
                f"{len(source)} sent, {len(target)} received"
            )
        return _unchanged


def _unchanged(data: Data) -> Data:
    return data


@CouplingMapping.register
class NearestNeighborMapping(CouplingMapping):
    """Maps every receiving point to the nearest sending point, found once: points must not move."""

    type: Literal["nearest-neighbor"] = "nearest-neighbor"

    def bind(self, source: Any, target: Any) -> BoundMapping:
        if source is None or target is None:
            if source is None and target is None:
                missing = "neither side has any: give both patches"
            else:
                side = "sending" if source is None else "receiving"
                missing = f"the {side} side has none: give its patch"
            raise ValueError(
                f"mapping '{self.type}' needs the coordinates of both sides, but {missing} "
                "coordinates (@coupled.coordinates) or choose mapping 'identity'"
            )
        indices = nearest_neighbor(source, target)

        def mapper(data: Data) -> Data:
            return {name: np.asarray(values)[indices] for name, values in data.items()}

        return mapper


def default_mapping() -> CouplingMapping:
    """The mapping of an exchange that names none."""
    return NearestNeighborMapping()


@PluginSystem.register(discriminator_variable="acceleration", discriminator="type")
class CouplingAcceleration(BaseModel):
    """Relaxes the data carried into a pass, per exchange; a plugin family.

    It keeps state across passes: give each exchange its own.
    """

    model_config = ConfigDict(extra="forbid")

    def __call__(self, sent: list[Data], iteration: int) -> list[Data]:
        """The relaxed ``sent`` for pass ``iteration``."""
        raise NotImplementedError


@CouplingAcceleration.register
class ConstantRelaxation(CouplingAcceleration):
    """``x = x_last + relaxation * (sent - x_last)``; first data passes unchanged."""

    type: Literal["constant"] = "constant"
    relaxation: float = Field(gt=0, le=1)
    _last: Optional[Any] = PrivateAttr(default=None)

    def __call__(self, sent: list[Data], iteration: int) -> list[Data]:
        values = _flatten(sent)
        if self._last is not None:
            residual = values - self._last
            values = self._last + self._factor(residual, iteration) * residual
        self._last = values
        return _unflatten(values, sent)

    def _factor(self, residual: Any, iteration: int) -> float:
        return self.relaxation


@CouplingAcceleration.register
class AitkenRelaxation(ConstantRelaxation):
    """Under-relaxes by Aitken's dynamic factor; for when a fixed factor converges slowly.

    The first pass of a time step uses ``relaxation``.

    Example::

        CouplingNode("n", ("i",), acceleration={"i": AitkenRelaxation(relaxation=0.5)})
    """

    type: Literal["aitken"] = "aitken"  # type: ignore[assignment]
    _residual: Optional[Any] = PrivateAttr(default=None)
    _omega: float = PrivateAttr(default=0.0)

    def _factor(self, residual: Any, iteration: int) -> float:
        previous = self._residual if iteration > 1 else None
        self._residual = residual
        # a round-off previous residual would drive the factor to zero and stall the loop
        if previous is None or np.linalg.norm(previous) <= 1e-6 * np.linalg.norm(residual):
            self._omega = self.relaxation
            return self._omega
        change = residual - previous
        squared = float(change @ change)
        if squared > 0.0:
            self._omega *= -float(previous @ change) / squared
        return self._omega


def _flatten(sent: list[Data]) -> Any:
    arrays = [np.ravel(data[name]) for data in sent for name in sorted(data)]
    return np.concatenate(arrays) if arrays else np.zeros(0)


def _unflatten(values: Any, like: list[Data]) -> list[Data]:
    result: list[Data] = []
    start = 0
    for data in like:
        unflattened: Data = {}
        for name in sorted(data):
            shape = np.shape(data[name])
            size = int(np.prod(shape))
            unflattened[name] = values[start : start + size].reshape(shape)
            start += size
        result.append(unflattened)
    return result


@PluginSystem.register(discriminator_variable="convergence", discriminator="type")
class CouplingConvergence(BaseModel):
    """Whether a node converged, from every side's change in the last pass; a plugin family.

    ``data`` names the field a measure compares, ``None`` for a measure of all fields.
    """

    model_config = ConfigDict(extra="forbid")
    data: Optional[str] = None

    def __call__(self, changes: list[SideChange]) -> bool:
        """Whether the data converged, from ``changes``."""
        raise NotImplementedError


@CouplingConvergence.register
class RelativeConvergence(CouplingConvergence):
    """Converged once every array of field ``data`` changes by less than ``limit`` (2-norms).

    Example::

        CouplingNode("n", ("interface",), convergence=(RelativeConvergence(data="T", limit=1e-5),))
    """

    type: Literal["relative"] = "relative"
    data: str
    limit: float = Field(gt=0)

    def __call__(self, changes: list[SideChange]) -> bool:
        return all(
            np.linalg.norm(np.asarray(change.current[name]) - np.asarray(change.previous[name]))
            <= self.limit * np.linalg.norm(change.current[name])
            for change in changes
            if change.field == self.data
            for name in change.current
        )
