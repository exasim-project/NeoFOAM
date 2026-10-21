# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

# NB: no ``from __future__ import annotations`` — Context injection matches
# ``param.annotation is Context`` literally (see field_writer.py).

"""postProcess — the core Model that evaluates a case's tables while it runs.

A solver-agnostic core Model, like ``fieldWriter``: LOAD reads the case's tables
(script and spec file), BUILD creates the :class:`PostProcessor` that runs them
(one write policy and one :class:`~neofoam.postprocess.writers.writer.TableWriter`
per table, both opened at the first write), and the model steps
``post_process`` last in the time loop — after the fields have been written, so
a CSV row and a time directory describe the same state. A case that declares no
table gets an empty set and the operation does nothing (not even a
``postProcessing/`` directory).
"""

from pathlib import Path
from typing import Any, NamedTuple, Protocol, runtime_checkable

from neofoam.algorithms.field_writer.write_control import WriteControl
from neofoam.framework.context import Context
from neofoam.framework.initialization import InitStep, model
from neofoam.framework.model import Model
from neofoam.postprocess.node import AggregatedDataSet, Pipeline
from neofoam.postprocess.table import TableSet, tables_for_case
from neofoam.postprocess.writers.writer import TableWriter

postProcess = Model("postProcess")

#: Where the CSVs land, relative to the case directory.
OUTPUT_DIR = "postProcessing"


@runtime_checkable
class _RunStart(Protocol):
    """A loop state that knows which time the run started from."""

    start_time: float


class _TableRuntime(NamedTuple):
    """The state one table needs while the run goes on: when to write, and where."""

    write_control: WriteControl
    writer: TableWriter


class PostProcessor:
    """The runtime state of the model: a private write policy and writer per table.

    Evaluates a table only on that table's own write steps, so a table with a
    coarse cadence costs nothing in between. Decomposed, every rank evaluates
    every due table — the aggregators reduce over the ranks — and only the
    master rank's writer touches the filesystem. Both copies are made on the
    first write, because only then is it known whether the run continues an
    earlier one (append) or starts from scratch (truncate)::

        PostProcessor(tables, case_dir / "postProcessing").write(ctx)
    """

    def __init__(self, tables: TableSet, base_path: Path) -> None:
        self._tables = tables.tables
        self._base_path = base_path
        self._runtime: dict[str, _TableRuntime] = {}

    def write(self, ctx: Context) -> None:
        """Evaluate every table due at this step and append its rows."""
        if not self._runtime:
            self._runtime = self._open(_continues_earlier_run(ctx.time))
        for table in self._tables:
            write_control, writer = self._runtime[table.name]
            if not write_control.should_write(ctx.time):
                continue
            result = table.pipeline.compute(ctx)
            if not isinstance(result, AggregatedDataSet):
                raise TypeError(
                    f"postProcess table {table.name!r}: the pipeline ends in "
                    f"{_final_step(table.pipeline)}, not in an aggregator, so there "
                    f"are no rows to write"
                )
            writer.write(ctx.time.value, result)

    def _open(self, append: bool) -> dict[str, _TableRuntime]:
        """A private copy of every table's write policy and writer, opened on its file.

        Both are shared specs — a ``TableSet``'s defaults serve every table that
        declared none — and both keep state between steps: a policy such as
        ``RunTimeWriteControl`` the time it last fired, a writer its file. The
        copies are what runs, so the declarations stay untouched and one table
        never consumes another's interval.
        """
        runtime: dict[str, _TableRuntime] = {}
        for table in self._tables:
            writer = table.writer.model_copy(deep=True)
            writer.open(self._base_path / table.name, append=append)
            runtime[table.name] = _TableRuntime(table.write_control.model_copy(deep=True), writer)
        return runtime


def _final_step(pipeline: Pipeline) -> str:
    """How to name a pipeline's last node when it turned out not to aggregate."""
    if not pipeline.steps:
        return "no nodes"
    return f"node {getattr(pipeline.steps[-1], 'type', type(pipeline.steps[-1]).__name__)!r}"


def _continues_earlier_run(step: Any) -> bool:
    """True for a restart (``start_time > 0``), whose CSVs are appended to."""
    return isinstance(step, _RunStart) and step.start_time > 0.0


# -- load: the case's two front doors, merged --------------------------------
@postProcess.load
def load(case_dir: Path, instance_id: str) -> TableSet:
    return tables_for_case(Path(case_dir))


# -- build: the PostProcessor is this model's runtime state ------------------
@postProcess.build
def build(tables: TableSet) -> list[InitStep]:
    # The case directory is only available from LOAD, so the TableSet carries it.
    base_path = Path(tables.case_dir or ".") / OUTPUT_DIR

    def create_processor(ctx: dict[str, Any]) -> PostProcessor:
        return PostProcessor(tables, base_path)

    return [model("post_processor", create_processor)]


# -- operation: append a row per due table -----------------------------------
# Stepped last in the solver's time loop; builder-insertion order already puts it
# after every field-mutating op and after the writer, so no ``depends_on``.
@postProcess.operation()
def post_process(self: Any, ctx: Context) -> None:
    """Evaluate the due tables against the live Context and append their rows."""
    processor: PostProcessor = ctx.models["post_processor"]
    processor.write(ctx)
