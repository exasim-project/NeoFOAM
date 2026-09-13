# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""CsvWriter — appending a table's aggregation results to one CSV file."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Literal

import pybFoam as pyf
from pydantic import PrivateAttr

from neofoam.postprocess.node import AggregatedDataSet
from neofoam.postprocess.writers.writer import TableWriter


def _owns_the_output_files() -> bool:
    """True on the rank that writes; in serial, always."""
    return bool(pyf.Pstream.master())


def _as_time_column(time: float) -> str:
    """The time as a reader expects to see it: ``0.05``, not ``0.049999999999999996``.

    An accumulated time is not the decimal the case file asked for, and its full
    repr in every row is unreadable. Twelve significant digits keep every time a
    run can resolve while dropping the accumulation noise.
    """
    return f"{time:.12g}"


@TableWriter.register
class CsvWriter(TableWriter):
    """Append the time-prefixed rows of an aggregation to one CSV file.

    The default writer (``{type: csv}``), taking one
    :class:`~neofoam.postprocess.node.AggregatedDataSet` per write step:
    ``time`` is the first column, the header (``time`` plus the result's
    headers) is written on the first row and the parent directory is created
    lazily, so a case with no tables leaves no
    ``postProcessing/`` behind. Open with ``append=True`` to continue an existing
    file (a restart) instead of truncating it::

        writer = CsvWriter()
        writer.open(case_dir / "postProcessing" / "volume_p", append=False)
        writer.write(0.1, result)

    Decomposed, every rank computes its table but only the master rank touches
    the filesystem, so a run leaves one set of CSVs and not one per processor.
    """

    type: Literal["csv"] = "csv"

    _path: Path = PrivateAttr(default=Path())
    _append: bool = PrivateAttr(default=False)
    _started: bool = PrivateAttr(default=False)

    def open(self, path_stem: Path, *, append: bool) -> None:
        """Bind this copy to ``<path_stem>.csv``; the file waits for the first row."""
        self._path = path_stem.with_name(path_stem.name + ".csv")
        self._append = append
        self._started = False

    def write(self, time: float, result: AggregatedDataSet) -> None:
        """Append one line per row of ``result``, prefixed by ``time``."""
        if not _owns_the_output_files():
            return
        if not self._started:
            self._start(result.headers)
        with self._path.open("a", newline="") as handle:
            writer = csv.writer(handle)
            for row in result.rows:
                writer.writerow([_as_time_column(time), *row])

    def _start(self, headers: list[str]) -> None:
        """Create the directory and write the header, unless continuing a file."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        continuing = self._append and self._path.exists()
        if not continuing:
            with self._path.open("w", newline="") as handle:
                csv.writer(handle).writerow(["time", *headers])
        self._started = True
