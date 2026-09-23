# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Binning nodes — group elements so an aggregator emits one row per group."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import field_validator

from neofoam.postprocess import _kernels as pp  # NeoFOAM post-processing kernels
from neofoam.postprocess.node import FieldDataSets, Node


@Node.register
class Directional(Node):
    """Bin elements by their signed distance along a direction.

    Put it before an aggregator to turn one row into a profile — one row per
    bin, ordered from below ``bins[0]`` to above ``bins[-1]``. ``direction`` is
    normalised, so ``bins`` are distances in metres and ``(2, 0, 0)`` bins
    exactly like ``(1, 0, 0)``. The row count is ``len(bins) + 1`` whatever the
    data contains, so every rank agrees on it::

        field("U") | Directional(bins=[0.02, 0.04], direction=(0, 1, 0)) | Mean()
    """

    type: Literal["directional"] = "directional"
    bins: list[float]
    direction: tuple[float, float, float]
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @field_validator("bins")
    @classmethod
    def _require_increasing_edges(cls, bins: list[float]) -> list[float]:
        # the kernel follows np.digitize, which silently switches to its
        # decreasing convention otherwise, and the profile comes out reversed
        # with no warning.
        if any(upper <= lower for lower, upper in zip(bins, bins[1:])):
            raise ValueError(f"directional: bins must be strictly increasing, got {bins}")
        return bins

    @field_validator("direction")
    @classmethod
    def _reject_the_zero_vector(
        cls, direction: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        if not np.linalg.norm(direction) > 0.0:
            raise ValueError("directional: direction must have a non-zero length")
        return direction

    def compute(self, dataset: FieldDataSets) -> FieldDataSets:
        # the kernel normalises the direction, so the edges are distances.
        groups = pp.bin_index(dataset.geometry.positions(), self.direction, self.origin, self.bins)
        return dataset.with_groups(groups, n_groups=len(self.bins) + 1)
