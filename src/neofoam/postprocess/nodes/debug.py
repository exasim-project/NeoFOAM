# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Debugging nodes — trivial steps that exercise and inspect a pipeline."""

from __future__ import annotations

from typing import Any, ClassVar, Literal

import numpy as np

from neofoam.postprocess.node import AggregatedDataSet, DataSet, Node


@Node.register
class Scale(Node):
    """Multiply the values by a factor — a trivial transform node.

    Use it to check that a table runs at all (or to convert a unit)::

        field("p") | Scale(factor=1000.0) | VolIntegrate()
    """

    type: Literal["scale"] = "scale"
    factor: float = 1.0

    def compute(self, dataset: DataSet) -> DataSet:
        return dataset.with_values(np.asarray(dataset.values) * self.factor)


@Node.register
class Print(Node):
    """Print a one-line summary of what flows through, then pass it on unchanged.

    A debugging aid to watch a pipeline run; works on a DataSet as well as on an
    aggregation, so it can be inserted anywhere::

        field("p") | Print(label="cells") | VolIntegrate() | Print()
    """

    type: Literal["print"] = "print"
    label: str = "postProcess"

    accepts_aggregated: ClassVar[bool] = True

    def compute(self, dataset: Any) -> Any:
        if isinstance(dataset, AggregatedDataSet):
            rows = [dict(zip(dataset.headers, row)) for row in dataset.rows]
            for columns in rows:
                print(f"[{self.label}] {dataset.name}: {columns}")
            if not rows:
                print(f"[{self.label}] {dataset.name}: no rows")
        else:
            values = np.asarray(dataset.values)
            # A plane can cut no cell on this rank; a debug node must not raise.
            extent = (
                f"min={values.min():.6g} max={values.max():.6g}" if values.size else "no elements"
            )
            print(f"[{self.label}] {dataset.name}: shape={values.shape} {extent}")
        return dataset
