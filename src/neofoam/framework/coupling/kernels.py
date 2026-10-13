# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""numpy kernels of a coupling: mapping, relaxation and convergence of the exchanged data.

Plug them into a :class:`~neofoam.framework.coupling.CouplingOp`::

    CouplingOp(
        converged=RelativeConvergence(data="T", limit=1e-5),
        transfer=MappedTransfer(),
        acceleration=AitkenRelaxation(initial=0.5),
    )
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .loop import CoupledSide, Data, SideChange


def nearest_neighbor(source: Any, target: Any) -> Any:
    """For every target point, the index of the nearest source point.

    Points are rows of ``(n_points, dim)`` arrays. Compares every pair of points,
    so it suits interfaces of a few thousand points.

    Example::

        values_on_target = values_on_source[nearest_neighbor(source, target)]
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    distances = np.linalg.norm(target[:, None, :] - source[None, :, :], axis=-1)
    return np.argmin(distances, axis=1)


class MappedTransfer:
    """Hands the data of one side to the other, mapped to its points by :func:`nearest_neighbor`.

    Use it as the ``transfer`` of a :class:`~neofoam.framework.coupling.CouplingOp`
    when the two sides do not share their points. It keeps the mapping of each
    pair of sides, whose points must not move. Not thread safe.

    Example::

        CouplingOp(transfer=MappedTransfer())
    """

    def __init__(self) -> None:
        self._indices: dict[tuple[int, int], Any] = {}

    def __call__(
        self, data: Data, source: CoupledSide, target: CoupledSide, iteration: int
    ) -> Data:
        """``data`` sent by ``source``, on the points of ``target``."""
        pair = (id(source), id(target))
        if pair not in self._indices:
            self._indices[pair] = nearest_neighbor(source.coordinates, target.coordinates)
        return {name: np.asarray(values)[self._indices[pair]] for name, values in data.items()}


class ConstantRelaxation:
    """Under-relaxes the data carried into a pass by a fixed factor.

    Use it as the ``acceleration`` of a :class:`~neofoam.framework.coupling.CouplingOp`:
    ``x = x_last + omega * (sent - x_last)`` for all carried data at once, with
    ``x_last`` what the last pass received. ``omega = 1`` changes nothing; the
    first data passes unchanged. Not thread safe.

    Example::

        CouplingOp(acceleration=ConstantRelaxation(omega=0.5))
    """

    def __init__(self, omega: float) -> None:
        self.omega = omega
        self._last: Optional[Any] = None

    def __call__(self, sent: list[Data], iteration: int) -> list[Data]:
        """The relaxed ``sent`` for pass ``iteration``."""
        values = _flatten(sent)
        if self._last is not None:
            residual = values - self._last
            values = self._last + self._factor(residual, iteration) * residual
        self._last = values
        return _unflatten(values, sent)

    def _factor(self, residual: Any, iteration: int) -> float:
        return self.omega


class AitkenRelaxation(ConstantRelaxation):
    """Under-relaxes the data carried into a pass by Aitken's dynamic factor.

    Use it instead of :class:`ConstantRelaxation` when a fixed factor converges
    slowly; on a linear fixed-point problem of one value it reaches the solution
    in the second relaxed pass. The factor follows from the residuals of the last
    two passes; the first pass of a time step uses ``initial``. Not thread safe.

    Example::

        CouplingOp(acceleration=AitkenRelaxation(initial=0.5))
    """

    def __init__(self, initial: float) -> None:
        super().__init__(initial)
        self._residual: Optional[Any] = None
        self._omega = initial

    def _factor(self, residual: Any, iteration: int) -> float:
        previous = self._residual if iteration > 1 else None
        self._residual = residual
        # A previous residual that is round-off against this one (e.g. a time step
        # that starts converged) would drive the factor to zero and stall the
        # loop: start over from the initial factor instead.
        if previous is None or np.linalg.norm(previous) <= 1e-6 * np.linalg.norm(residual):
            self._omega = self.omega
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


class RelativeConvergence:
    """Converged once the data of field ``data`` changes by less than ``limit`` relative to it.

    Use it as the ``converged`` of a :class:`~neofoam.framework.coupling.CouplingOp`.
    It checks every array sent by a side of the field ``data`` (2-norm of the
    change over the 2-norm of the array) and ignores the sides of other fields.

    Example::

        CouplingOp(converged=RelativeConvergence(data="T", limit=1e-5))
    """

    def __init__(self, data: str, limit: float) -> None:
        self.data = data
        self.limit = limit

    def __call__(self, changes: list[SideChange]) -> bool:
        """Whether every array of field ``data`` changed by less than ``limit`` in the last pass."""
        return all(
            np.linalg.norm(np.asarray(change.current[name]) - np.asarray(change.previous[name]))
            <= self.limit * np.linalg.norm(change.current[name])
            for change in changes
            if change.field == self.data
            for name in change.current
        )
