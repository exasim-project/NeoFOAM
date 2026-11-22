# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Spec for the TableWriter plugin family: a table picks the format it is written in.

A writer is an extension point like a node: a case's ``system/postProcess.py``
registers one with ``@TableWriter.register`` and the same case's spec file then
selects it by ``type`` — which is the whole point of the family, since NeoFOAM
ships only the CSV writer. The cases are checked in (``cases/custom_writer``,
``cases/unknown_writer``) and copied into ``tmp_path``, because executing a
script writes bytecode next to it. Registration is process-wide, so the fixture
puts the registries back. No OpenFOAM: a table is only built here, never
evaluated.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterator

import pytest

from neofoam.core.plugin_system import PluginSystem
from neofoam.postprocess.config import PostProcessConfig, resolve_table
from neofoam.postprocess.table import tables_for_case
from neofoam.postprocess.writers.csv import CsvWriter

CASES = Path(__file__).parent.parent / "cases"


@pytest.fixture(autouse=True)
def restore_plugin_registries() -> Iterator[None]:
    """Unregister what a case script registered, so the next test sees a clean union."""
    before = {family: list(plugins) for family, plugins in PluginSystem.list_plugins().items()}
    yield
    for family, plugins in PluginSystem.list_plugins().items():
        for plugin_cls in list(plugins):
            if plugin_cls not in before.get(family, []):
                PluginSystem.remove_plugin_model(family, plugin_cls)


def _case(tmp_path: Path, name: str) -> Path:
    """A private copy of a checked-in case — executing its script writes bytecode."""
    case_dir = tmp_path / name
    shutil.copytree(CASES / name, case_dir)
    return case_dir


def test_table_without_a_declared_writer_is_written_as_csv() -> None:
    spec = PostProcessConfig.load(case_dir=CASES / "one_table/system/postProcess.yaml").tables[0]

    assert isinstance(resolve_table(spec).writer, CsvWriter)


def test_a_writer_the_script_registers_is_usable_by_type_in_the_same_case(tmp_path: Path) -> None:
    tables = tables_for_case(_case(tmp_path, "custom_writer"))

    by_name = {table.name: table for table in tables.tables}
    assert set(by_name) == {"volume_p_script", "volume_p_declared"}
    declared, scripted = by_name["volume_p_declared"], by_name["volume_p_script"]
    assert type(declared.writer) is type(scripted.writer)


def test_unknown_writer_type_names_the_table_and_the_types_that_would_work() -> None:
    spec = PostProcessConfig.load(case_dir=CASES / "unknown_writer/system/postProcess.yaml").tables[
        0
    ]

    with pytest.raises(ValueError, match=r"'volume_p'.*nope.*registered writer types:.*csv"):
        resolve_table(spec)


def test_loading_the_same_script_twice_still_resolves_its_writer(tmp_path: Path) -> None:
    # the re-load must take the first generation's registration back, or the two
    # classes sharing ``type: text`` make the family's union ambiguous
    case_dir = _case(tmp_path, "custom_writer")
    first = tables_for_case(case_dir)

    second = tables_for_case(case_dir)

    assert [table.name for table in second.tables] == [table.name for table in first.tables]
