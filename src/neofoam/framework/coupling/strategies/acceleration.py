# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Accelerations of the data carried into a coupling iteration: :class:`CouplingAcceleration`."""

from __future__ import annotations

from typing import Any, Literal, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from neofoam.core.plugin_system import PluginSystem

from ..comm.channel import Data


@PluginSystem.register(discriminator_variable="acceleration", discriminator="type")
class CouplingAcceleration(BaseModel):
    """Relaxes the data carried into an iteration, per exchange; a plugin family.

    It keeps state across iterations: give each exchange its own.
    """

    model_config = ConfigDict(extra="forbid")

    def __call__(self, sent: list[Data], iteration: int) -> list[Data]:
        """The relaxed ``sent`` for iteration ``iteration``."""
        raise NotImplementedError


@CouplingAcceleration.register
class ConstantRelaxation(CouplingAcceleration):
    """``x = x_last + relaxation * (sent - x_last)``.

    The first data of every time step (iteration 1) passes unchanged.
    """

    type: Literal["constant"] = "constant"
    relaxation: float = Field(gt=0, le=1)
    _last: Optional[Any] = PrivateAttr(default=None)

    def __call__(self, sent: list[Data], iteration: int) -> list[Data]:
        values = _flatten(sent)
        if iteration == 1:
            self._last = None
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

    The first iteration of a time step uses ``relaxation``.

    Example::

        CouplingLoop("n", ("i",), acceleration={"i": AitkenRelaxation(relaxation=0.5)})
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
