# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Rows — the terminal node that writes every element instead of reducing them."""

from __future__ import annotations

from typing import Any, Literal, Optional

import numpy as np

from neofoam.postprocess.node import AggregatedDataSet, DataSet, Node


@Node.register
class Rows(Node):
    """One CSV row per active element: its position, then its value.

    The terminal node of a probe table — a write step appends as many lines as
    the source has active elements, with columns ``x,y,z,<name>`` (``<name>_0``
    to ``<name>_2`` for a vector). Use an aggregator instead when one number per
    write step is wanted; masked-out elements (a sample point outside the mesh,
    a selector upstream) are left out entirely::

        line("U", start, end, 20) | Mag() | Rows(name="U_profile")
    """

    type: Literal["rows"] = "rows"
    name: Optional[str] = None

    def compute(self, dataset: DataSet) -> AggregatedDataSet:
        values = np.asarray(dataset.values, dtype=float)
        components = values if values.ndim > 1 else values[:, None]
        positions = np.asarray(dataset.geometry.positions, dtype=float)
        active: Any = slice(None) if dataset.mask is None else np.asarray(dataset.mask, dtype=bool)
        label = self.name or dataset.name
        n_components = components.shape[1]
        columns = [label] if n_components == 1 else [f"{label}_{i}" for i in range(n_components)]
        grouped = dataset.groups is not None
        bins = np.asarray(dataset.groups, dtype=np.int64) if grouped else np.zeros(len(values))
        return AggregatedDataSet(
            name=label,
            headers=[*(["bin"] if grouped else []), "x", "y", "z", *columns],
            rows=[
                [*([float(group)] if grouped else []), *position, *value]
                for group, position, value in zip(
                    bins[active], positions[active], components[active]
                )
            ],
        )
