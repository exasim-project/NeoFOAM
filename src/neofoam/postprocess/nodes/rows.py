# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Rows — the terminal node that writes every element instead of reducing them."""

from __future__ import annotations

from typing import Any, Literal, Optional

import numpy as np

from neofoam.postprocess._reduce import _is_parallel_run
from neofoam.postprocess.node import AggregatedData, AggregatedDataSet, FieldDataSets, Node
from neofoam.postprocess.nodes._arrays import host
from neofoam.postprocess.nodes.aggregators import BIN_COLUMN

#: The columns an element's position fills, before its value.
POSITION_COLUMNS = ["x", "y", "z"]


@Node.register
class Rows(Node):
    """One CSV row per active element: its position, then its value.

    The terminal node of a probe table — a write step appends as many lines as
    the source has active elements, with columns ``x,y,z,<name>`` (``<name>_0``
    to ``<name>_2`` for a vector). Use an aggregator instead when one number per
    write step is wanted; masked-out elements (a sample point outside the mesh,
    a selector upstream) are left out entirely::

        line("U", start, end, 20) | Mag() | Rows(name="U_profile")

    It writes the elements themselves, so this is the one node that copies a
    NeoN field off its executor — a row per cell is host data by definition.

    Serial only: an aggregator reduces its per-bin numbers over the ranks, but a
    row table would need the elements themselves gathered onto the master, and
    pybFoam binds no gather. Rather than write the master's share and silently
    drop every other rank's points, a decomposed run raises here.
    """

    type: Literal["rows"] = "rows"
    name: Optional[str] = None

    def compute(self, dataset: FieldDataSets) -> AggregatedDataSet:
        if _is_parallel_run():
            raise NotImplementedError(
                f"postProcess: the rows table {self.name or dataset.name!r} cannot run "
                "decomposed — its elements live on every rank and pybFoam binds no gather "
                "onto the master, so only the master rank's points would be written. Use an "
                "aggregator (sum, mean, max, min, volIntegrate, surfIntegrate), which does "
                "reduce over the ranks, or run the case serially."
            )
        values = host(dataset.field).astype(float)
        positions = host(dataset.geometry.positions()).astype(float)
        active: Any = slice(None) if dataset.mask is None else host(dataset.mask).astype(bool)
        grouped = dataset.groups is not None
        bins = host(dataset.groups).astype(float) if grouped else np.zeros(len(values))
        group_name = [*([BIN_COLUMN] if grouped else []), *POSITION_COLUMNS]
        return AggregatedDataSet(
            name=self.name or dataset.name,
            values=[
                AggregatedData(
                    value=float(value) if values.ndim == 1 else [float(entry) for entry in value],
                    group=[*([float(group)] if grouped else []), *(float(x) for x in position)],
                    group_name=group_name,
                )
                for group, position, value in zip(bins[active], positions[active], values[active])
            ],
        )
