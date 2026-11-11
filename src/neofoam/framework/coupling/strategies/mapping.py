# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Mappings of the exchanged data onto the receiving port's location: :class:`CouplingMapping`."""

from __future__ import annotations

from typing import Any, Callable, ClassVar, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from neofoam.core.plugin_system import PluginSystem

from ..comm.channel import Data
from ..location import Geometry

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


# a mapping bound to both ports' geometries: data on the source -> on the target
BoundMapping = Callable[[Data], Data]


# the entity kinds of the data models: face, cell, value, probe
_TRANSFER = (("face", "face"), ("cell", "cell"))


@PluginSystem.register(discriminator_variable="mapping", discriminator="type")
class CouplingMapping(BaseModel):
    """Maps the data a port sends onto the receiving port's location; a plugin family.

    Bound once, when the receiving op connects, to the sender's geometry
    (``source``) and its own (``target``). ``kinds`` lists the ``(sent, received)``
    entity kinds it maps (``face``, ``cell``, ``value``, ``probe``). A mapping of
    one's own subclasses it with a ``type: Literal[...]``, overrides :meth:`bind`
    and registers (``@CouplingMapping.register``); so do accelerations and convergences.
    """

    model_config = ConfigDict(extra="forbid")
    kinds: ClassVar[tuple[tuple[str, str], ...]] = _TRANSFER

    def check(self, sent: str, received: str) -> None:
        """Raise ``ValueError`` unless the mapping maps ``sent`` onto ``received`` entities."""
        if (sent, received) not in self.kinds:
            can = ", ".join(f"{a} -> {b}" for a, b in self.kinds)
            raise ValueError(
                f"mapping '{getattr(self, 'type', type(self).__name__)}' maps {can}, "
                f"not {sent} -> {received}"
            )

    def bind(self, source: Geometry, target: Geometry) -> BoundMapping:
        """The mapping from ``source`` onto ``target``; ``ValueError`` when it cannot map."""
        raise NotImplementedError


@CouplingMapping.register
class IdentityMapping(CouplingMapping):
    """Hands the data over unchanged: the ports share their entities."""

    type: Literal["identity"] = "identity"
    kinds: ClassVar[tuple[tuple[str, str], ...]] = (
        *_TRANSFER,
        ("value", "value"),
        ("probe", "probe"),
    )

    def bind(self, source: Geometry, target: Geometry) -> BoundMapping:
        if source.size is not None and target.size is not None and source.size != target.size:
            raise ValueError(
                f"mapping '{self.type}' needs as many entities on both ports: "
                f"{source.size} sent, {target.size} received"
            )
        return _unchanged


def _unchanged(data: Data) -> Data:
    return data


def _points(mapping: str, source: Geometry, target: Geometry) -> tuple[Any, Any]:
    """Both ports' points; ``ValueError`` naming the port without."""
    if source.points is None or target.points is None:
        if source.points is None and target.points is None:
            missing = "neither port has any"
        else:
            missing = f"the {'sending' if source.points is None else 'receiving'} port has none"
        raise ValueError(
            f"mapping '{mapping}' needs the points of both ports, but {missing}: give its "
            "mesh geometry or choose mapping 'identity'"
        )
    return source.points, target.points


def _taken(indices: Any) -> BoundMapping:
    def mapper(data: Data) -> Data:
        return {name: np.asarray(values)[indices] for name, values in data.items()}

    return mapper


@CouplingMapping.register
class NearestNeighborMapping(CouplingMapping):
    """Maps every receiving entity to the nearest sending one, found once: points must not move."""

    type: Literal["nearest-neighbor"] = "nearest-neighbor"

    def bind(self, source: Geometry, target: Geometry) -> BoundMapping:
        return _taken(nearest_neighbor(*_points(self.type, source, target)))


def _scalars(mapping: str, data: Data) -> Data:
    """``data`` as 1-d arrays; ``ValueError`` for a vector field."""
    arrays = {name: np.asarray(values, dtype=float) for name, values in data.items()}
    for name, values in arrays.items():
        if values.ndim != 1:
            raise ValueError(
                f"mapping '{mapping}' takes scalar fields; {name} has the shape {values.shape}"
            )
    return arrays


def _measures(mapping: str, source: Geometry, what: str) -> Any:
    if source.measures is None:
        raise ValueError(f"mapping '{mapping}' needs the {what} of the sending port; it has none")
    return source.measures


class _Reduction(CouplingMapping):
    """Reduces the values of every entity to one, weighted by their measures."""

    mean: ClassVar[bool] = False
    measure: ClassVar[str] = ""

    def bind(self, source: Geometry, target: Geometry) -> BoundMapping:
        name = getattr(self, "type")
        weights = np.asarray(_measures(name, source, self.measure), dtype=float)
        total = float(weights.sum()) if self.mean else 1.0

        def mapper(data: Data) -> Data:
            return {
                field: np.array(float(values @ weights) / total)
                for field, values in _scalars(name, data).items()
            }

        return mapper


@CouplingMapping.register
class AreaMeanMapping(_Reduction):
    """The area-weighted mean of a patch's values: ``sum(q A) / sum(A)``."""

    type: Literal["area-mean"] = "area-mean"
    kinds: ClassVar[tuple[tuple[str, str], ...]] = (("face", "value"),)
    mean: ClassVar[bool] = True
    measure: ClassVar[str] = "face areas"


@CouplingMapping.register
class AreaIntegralMapping(_Reduction):
    """The integral of a patch's values over its faces: ``sum(q A)``."""

    type: Literal["area-integral"] = "area-integral"
    kinds: ClassVar[tuple[tuple[str, str], ...]] = (("face", "value"),)
    measure: ClassVar[str] = "face areas"


@CouplingMapping.register
class VolumeMeanMapping(_Reduction):
    """The volume-weighted mean of the cells' values: ``sum(s V) / sum(V)``."""

    type: Literal["volume-mean"] = "volume-mean"
    kinds: ClassVar[tuple[tuple[str, str], ...]] = (("cell", "value"),)
    mean: ClassVar[bool] = True
    measure: ClassVar[str] = "cell volumes"


@CouplingMapping.register
class VolumeIntegralMapping(_Reduction):
    """The integral of the cells' values over their volumes: ``sum(s V)``."""

    type: Literal["volume-integral"] = "volume-integral"
    kinds: ClassVar[tuple[tuple[str, str], ...]] = (("cell", "value"),)
    measure: ClassVar[str] = "cell volumes"


@CouplingMapping.register
class UniformMapping(CouplingMapping):
    """One value on every face or cell of the receiving port."""

    type: Literal["uniform"] = "uniform"
    kinds: ClassVar[tuple[tuple[str, str], ...]] = (("value", "face"), ("value", "cell"))

    def bind(self, source: Geometry, target: Geometry) -> BoundMapping:
        size = target.size
        if size is None:
            raise ValueError(f"mapping '{self.type}' needs the size of the receiving port")

        def mapper(data: Data) -> Data:
            arrays = {name: np.asarray(values, dtype=float) for name, values in data.items()}
            for name, values in arrays.items():
                if values.ndim != 0:
                    raise ValueError(
                        f"mapping '{self.type}' takes scalar values; {name} has the shape "
                        f"{values.shape}"
                    )
            return {name: np.full(size, float(values)) for name, values in arrays.items()}

        return mapper


@CouplingMapping.register
class ProbeMapping(CouplingMapping):
    """The values of the cells nearest to ``points``, one row per point (no interpolation).

    Example::

        mapping: {type: probe, points: [[0.1, 0.01, 0], [0.2, 0.01, 0]]}
    """

    type: Literal["probe"] = "probe"
    kinds: ClassVar[tuple[tuple[str, str], ...]] = (("cell", "probe"),)
    points: list[tuple[float, ...]] = Field(min_length=1)

    def bind(self, source: Geometry, target: Geometry) -> BoundMapping:
        points = np.asarray(self.points, dtype=float)
        return _taken(nearest_neighbor(*_points(self.type, source, Geometry.of(points, None))))


def default_mapping() -> CouplingMapping:
    """The mapping of an exchange that names none."""
    return NearestNeighborMapping()
