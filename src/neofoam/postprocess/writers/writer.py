# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The TableWriter plugin family: the output format one table is written in."""

from __future__ import annotations

from pathlib import Path

from pydantic import ConfigDict

from neofoam.core.plugin_system import PluginSystem
from neofoam.io import BaseConfig
from neofoam.postprocess.node import AggregatedDataSet


@PluginSystem.register(discriminator_variable="writer", discriminator="type")
class TableWriter(BaseConfig):
    """Plugin interface: where a table's aggregation results are written.

    Named for the table and not just ``Writer`` because
    :class:`neofoam.algorithms.field_writer.writer.Writer` already holds that
    name, and :class:`~neofoam.core.plugin_system.PluginSystem` keys a family by
    its class name — two families sharing one would silently share a registry.

    Add a format with a ``@TableWriter.register`` subclass carrying a
    ``type: Literal[...]`` discriminator and the two methods below; a table then
    selects it by that string (spec file ``writer: {type: ...}``, script
    ``@postProcess.table(..., writer=...)``), defaulting to
    :class:`~neofoam.postprocess.writers.csv.CsvWriter`. Unknown keys are
    refused, so a case file's typo is an error and not a default.

    There is no ``close``: the framework has no end-of-run hook, so a format
    that buffers has to flush inside :meth:`write` as
    :class:`~neofoam.postprocess.writers.csv.CsvWriter` does (it opens and
    closes the file per write).

    A writer held by a :class:`~neofoam.postprocess.table.Table` is a *spec* and
    is never written through — the file state a writer keeps between calls (its
    path, whether the header is out) belongs to the private copy
    :class:`~neofoam.postprocess.model.PostProcessor` deep-copies and opens per
    table, so one spec can serve every table of a case::

        writer = CsvWriter().model_copy(deep=True)
        writer.open(case_dir / "postProcessing" / "volume_p", append=False)
        writer.write(0.1, result)

    ``path_stem`` carries no suffix: the writer appends the one its format owns.
    """

    model_config = ConfigDict(extra="forbid")

    def open(self, path_stem: Path, *, append: bool) -> None:
        """Bind this copy to its output file; ``append`` continues a restart's file."""
        raise NotImplementedError

    def write(self, time: float, result: AggregatedDataSet) -> None:
        """Append one entry per row of ``result``, prefixed by ``time``."""
        raise NotImplementedError
