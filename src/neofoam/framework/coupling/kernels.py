# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The strategies of a coupling: mapping, acceleration and convergence of the exchanged data.

Each is a :class:`~neofoam.core.plugin_system.PluginSystem` family, a strategy
selected by its ``type``. A strategy of one's own subclasses its family with a
``type: Literal[...]`` and registers (``@CouplingMapping.register``)::

    CouplingNode(
        "thermal",
        ("interface",),
        convergence=(RelativeConvergence(data="T", limit=1e-5),),
        acceleration={"interface": AitkenRelaxation(relaxation=0.5)},
    )
"""

from __future__ import annotations

from typing import Any, Callable, Literal, NamedTuple, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from neofoam.core.plugin_system import PluginSystem

from .transport import Data


class SideChange(NamedTuple):
    """The :data:`~neofoam.framework.coupling.Array` data one side sent in the previous and
    in the current pass.

    For a convergence check; plain fields are not compared.
    """

    field: str
    previous: Data
    current: Data


# The elements of the target-by-source distance block :func:`nearest_neighbor` computes at once.
_BLOCK_ELEMENTS = 1 << 20


def nearest_neighbor(source: Any, target: Any) -> Any:
    """For every target point, the index of the nearest source point (the lowest on a tie).

    Points are rows of ``(n_points, dim)`` arrays. Compares every pair of points,
    a block of target points at a time, so that memory grows with the points of
    one side, not with the pairs: O(n + m) for ``m`` source and ``n`` target
    points, time O(n * m).

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


#: A mapping bound to the points of both sides: the data on the source points -> on the target's.
BoundMapping = Callable[[Data], Data]


@PluginSystem.register(discriminator_variable="mapping", discriminator="type")
class CouplingMapping(BaseModel):
    """Maps the :data:`~neofoam.framework.coupling.Array` data a side sends onto the other side.

    The mapping of an :class:`~neofoam.framework.coupling.data.Exchange`, a plugin
    family (``system/regionCoupling.yaml``: ``mapping:``, its ``type`` and
    options). The receiving side's coupling op binds it once, when it connects,
    to the partner's points (``source``) and the side's own (``target``; each a
    ``(n_points, dim)`` array, or ``None`` without coordinates), and maps the
    data of every pass with the bound mapping. A mapping of one's own subclasses
    it with a ``type: Literal[...]``, its options as fields, overrides
    :meth:`bind` and registers (``@CouplingMapping.register``).

    Example::

        mapper = exchange.mapping.bind(partner_points, points)
        mapped = mapper({"T": values})
    """

    model_config = ConfigDict(extra="forbid")

    def bind(self, source: Any, target: Any) -> BoundMapping:
        """The mapping from the points ``source`` onto ``target``.

        Raises ``ValueError`` when it cannot map between them.
        """
        raise NotImplementedError


@CouplingMapping.register
class IdentityMapping(CouplingMapping):
    """Hands the data over unchanged: the sides share their points."""

    type: Literal["identity"] = "identity"

    def bind(self, source: Any, target: Any) -> BoundMapping:
        """Raises ``ValueError`` when both sides have points and their numbers differ."""
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
    """Maps every point of the receiving side to the nearest point of the sending side.

    Both sides need coordinates. :meth:`bind` finds the nearest points
    (:func:`nearest_neighbor`) once, so the points must not move.
    """

    type: Literal["nearest-neighbor"] = "nearest-neighbor"

    def bind(self, source: Any, target: Any) -> BoundMapping:
        """Raises ``ValueError`` naming the side without coordinates."""
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
    """The mapping of an exchange that names none: :class:`NearestNeighborMapping`.

    The one default of both a :class:`~neofoam.framework.coupling.region.Link`
    and an exchange of ``system/regionCoupling.yaml``.
    """
    return NearestNeighborMapping()


@PluginSystem.register(discriminator_variable="acceleration", discriminator="type")
class CouplingAcceleration(BaseModel):
    """Relaxes the Array data carried into a pass: (data of the carried sides, iteration) -> data.

    The acceleration of an exchange of a
    :class:`~neofoam.framework.coupling.CouplingNode`, a plugin family
    (``system/regionCoupling.yaml``: ``acceleration:``, its ``type`` and
    options). It keeps the data of its last passes in private attributes: give
    each exchange its own. An acceleration of one's own subclasses it with a
    ``type: Literal[...]``, its options as fields, overrides :meth:`__call__`
    and registers (``@CouplingAcceleration.register``).
    """

    model_config = ConfigDict(extra="forbid")

    def __call__(self, sent: list[Data], iteration: int) -> list[Data]:
        """The relaxed ``sent`` for pass ``iteration``."""
        raise NotImplementedError


@CouplingAcceleration.register
class ConstantRelaxation(CouplingAcceleration):
    """Under-relaxes the data carried into a pass by a fixed factor, ``relaxation``.

    ``x = x_last + relaxation * (sent - x_last)`` for all carried :data:`Array`
    data at once, with ``x_last`` what the last pass received. ``relaxation = 1``
    changes nothing; the first data passes unchanged. Not thread safe.

    Example::

        CouplingNode(
            "n", ("interface",), acceleration={"interface": ConstantRelaxation(relaxation=0.5)}
        )
    """

    type: Literal["constant"] = "constant"
    relaxation: float = Field(gt=0, le=1)
    _last: Optional[Any] = PrivateAttr(default=None)

    def __call__(self, sent: list[Data], iteration: int) -> list[Data]:
        """The relaxed ``sent`` for pass ``iteration``."""
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
    """Under-relaxes the data carried into a pass by Aitken's dynamic factor.

    Use it instead of :class:`ConstantRelaxation` when a fixed factor converges
    slowly; on a linear fixed-point problem of one value it reaches the solution
    in the second relaxed pass. The factor follows from the residuals of the last
    two passes; the first pass of a time step uses ``relaxation``. Not thread safe.

    Example::

        CouplingNode(
            "n", ("interface",), acceleration={"interface": AitkenRelaxation(relaxation=0.5)}
        )
    """

    type: Literal["aitken"] = "aitken"  # type: ignore[assignment]
    _residual: Optional[Any] = PrivateAttr(default=None)
    _omega: float = PrivateAttr(default=0.0)

    def _factor(self, residual: Any, iteration: int) -> float:
        previous = self._residual if iteration > 1 else None
        self._residual = residual
        # A previous residual that is round-off against this one (e.g. a time step
        # that starts converged) would drive the factor to zero and stall the
        # loop: start over from the initial factor instead.
        if previous is None or np.linalg.norm(previous) <= 1e-6 * np.linalg.norm(residual):
            self._omega = self.relaxation
            return self._omega
        change = residual - previous
        squared = float(change @ change)
        if squared > 0.0:
            self._omega *= -float(previous @ change) / squared
        return self._omega


def _flatten(sent: list[Data]) -> Any:
    """All arrays of ``sent`` as one vector, in order."""
    arrays = [np.ravel(data[name]) for data in sent for name in sorted(data)]
    return np.concatenate(arrays) if arrays else np.zeros(0)


def _unflatten(values: Any, like: list[Data]) -> list[Data]:
    """``values`` cut into the arrays of ``like``."""
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
    """Whether the coupling has converged, from the change of every side's data in the last pass.

    A convergence measure of a :class:`~neofoam.framework.coupling.CouplingNode`,
    a plugin family (``system/regionCoupling.yaml``: ``convergence:``, a list of
    ``type`` and options). A measure on the data of one field names it in its
    ``data`` option. A measure of one's own subclasses it with a
    ``type: Literal[...]``, its options as fields, overrides :meth:`__call__` and
    registers (``@CouplingConvergence.register``).
    """

    model_config = ConfigDict(extra="forbid")

    def __call__(self, changes: list[SideChange]) -> bool:
        """Whether the data converged, from ``changes``."""
        raise NotImplementedError


@CouplingConvergence.register
class RelativeConvergence(CouplingConvergence):
    """Converged once the data of field ``data`` changes by less than ``limit`` relative to it.

    It checks every array sent by a side of the field ``data`` (2-norm of the
    change over the 2-norm of the array) and ignores the sides of other fields.

    Example::

        CouplingNode("n", ("interface",), convergence=(RelativeConvergence(data="T", limit=1e-5),))
    """

    type: Literal["relative"] = "relative"
    data: str
    limit: float = Field(gt=0)

    def __call__(self, changes: list[SideChange]) -> bool:
        """Whether every array of field ``data`` changed by less than ``limit`` in the last pass."""
        return all(
            np.linalg.norm(np.asarray(change.current[name]) - np.asarray(change.previous[name]))
            <= self.limit * np.linalg.norm(change.current[name])
            for change in changes
            if change.field == self.data
            for name in change.current
        )
