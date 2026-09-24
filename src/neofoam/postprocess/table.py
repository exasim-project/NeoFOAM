# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Table and TableSet — a named pipeline with its cadence, and a case's set of them.

A :class:`Table` is what one CSV file contains; a :class:`TableSet` is what one
case declares, no matter which front door declared it: the decorator a case
script uses and the result :func:`tables_for_case` assembles from script *and*
spec file are the same object.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from neofoam.algorithms.field_writer.write_control import IntervalWriteControl, WriteControl
from neofoam.postprocess.node import Pipeline
from neofoam.postprocess.writers import CsvWriter, TableWriter


@dataclass(frozen=True)
class Table:
    """One output file: a named pipeline, when to evaluate it and what to write it as.

    Frozen because a table is a declaration, evaluated once per write step and
    never edited afterwards — the ``write_control`` and the ``writer`` are pure
    specs too, which is why :class:`~neofoam.postprocess.model.PostProcessor`
    runs private copies of them rather than these. Build it through the
    :meth:`TableSet.table` decorator or
    :func:`~neofoam.postprocess.config.resolve_table`.
    """

    name: str
    pipeline: Pipeline
    write_control: WriteControl
    writer: TableWriter


class TableSet:
    """The tables of one case — a script's decorator registry and the load result.

    A case script creates exactly one at module level and decorates its pipeline
    functions with :meth:`table`; the function runs at decoration time, so a
    script table *is* a :class:`~neofoam.postprocess.node.Pipeline`, exactly like
    a declared one::

        postProcess = TableSet()

        @postProcess.table("volume_p.csv")
        def volume_p():
            return field("p") | VolIntegrate(name="volume_p")

    ``write_control`` sets the cadence every table without its own inherits
    (default: every step) and ``writer`` the output format (default: CSV);
    ``case_dir`` is filled by :func:`tables_for_case` so the model knows where to
    write.
    """

    def __init__(
        self,
        write_control: Optional[WriteControl] = None,
        *,
        writer: Optional[TableWriter] = None,
        case_dir: Optional[Path] = None,
    ) -> None:
        self._default_write_control = write_control or IntervalWriteControl()
        self._default_writer = writer or CsvWriter()
        self.case_dir = case_dir
        self.tables: list[Table] = []

    def table(
        self,
        filename: str,
        *,
        write_control: Optional[WriteControl] = None,
        writer: Optional[TableWriter] = None,
    ) -> Callable[[Callable[[], Pipeline]], Callable[[], Pipeline]]:
        """Register the decorated function's pipeline as ``<filename>``'s table."""

        def decorator(build_pipeline: Callable[[], Pipeline]) -> Callable[[], Pipeline]:
            self.add(
                Table(
                    name=Path(filename).stem,
                    pipeline=build_pipeline(),
                    write_control=write_control or self._default_write_control,
                    writer=writer or self._default_writer,
                )
            )
            return build_pipeline

        return decorator

    def add(self, table: Table) -> None:
        """Append an already-built table."""
        self.tables.append(table)


def tables_for_case(case_dir: Path) -> TableSet:
    """Every table a case declares: the script's first, then the spec file's.

    The script runs first so the nodes it registers with ``@Node.register`` are
    selectable by ``type`` from the spec file. A name declared by both front
    doors raises, naming both origins. Neither file present ⇒ an empty set (the
    ``postProcess`` model then does nothing)::

        tables_for_case(Path("cases/cavity")).tables
    """
    from neofoam.postprocess.config import (  # noqa: PLC0415  # cycle: config->table
        load_config,
        resolve_table,
        spec_file,
    )
    from neofoam.postprocess.script import (  # noqa: PLC0415  # cycle: script->table
        SCRIPT_FILE,
        load_script,
    )

    case_dir = Path(case_dir)
    tables = TableSet(case_dir=case_dir)
    origins: dict[str, str] = {}

    script = load_script(case_dir)
    if script is not None:
        _extend(tables, origins, script.tables, SCRIPT_FILE)

    spec_path = spec_file(case_dir)
    if spec_path is not None:
        declared = [resolve_table(spec) for spec in load_config(case_dir).tables]
        _extend(tables, origins, declared, str(spec_path.relative_to(case_dir)))

    return tables


def _extend(target: TableSet, origins: dict[str, str], tables: list[Table], origin: str) -> None:
    """Add ``tables`` to ``target``, rejecting a name another origin already used."""
    for table in tables:
        previous = origins.get(table.name)
        if previous is not None:
            raise ValueError(
                f"postProcess: duplicate table {table.name!r} — declared in "
                f"{previous} and in {origin}"
            )
        origins[table.name] = origin
        target.add(table)
