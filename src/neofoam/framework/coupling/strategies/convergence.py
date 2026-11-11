# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Convergence measures of a coupling loop: :class:`CouplingConvergence`."""

from __future__ import annotations

from typing import Literal, NamedTuple, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from neofoam.core.plugin_system import PluginSystem

from ..comm.channel import Data


class SideChange(NamedTuple):
    """What one stream sent in the previous and in the current pass, for a convergence check."""

    previous: Data
    current: Data


@PluginSystem.register(discriminator_variable="convergence", discriminator="type")
class CouplingConvergence(BaseModel):
    """Whether a node converged, from every stream's change in the last pass; a plugin family.

    ``data`` names the data field a measure compares (``T``), ``None`` for all of them.
    """

    model_config = ConfigDict(extra="forbid")
    data: Optional[str] = None

    def __call__(self, changes: list[SideChange]) -> bool:
        """Whether the data converged, from ``changes``."""
        raise NotImplementedError


@CouplingConvergence.register
class RelativeConvergence(CouplingConvergence):
    """Converged once every array named ``data`` changes by less than ``limit`` (2-norms).

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
            for name in change.current
            if name == self.data
        )
