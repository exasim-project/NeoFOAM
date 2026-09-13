# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Spec for the script front door: executing a case's system/postProcess.py.

The scripts are checked-in case files (``cases/*/system/postProcess.py``) —
exactly what a user writes — because executing them *is* the behaviour under
test. Two properties carry the design: a script table is the same
:class:`~neofoam.postprocess.node.Pipeline` a spec file resolves to (that is why
the decorated function takes no mesh), and a node the script registers is
selectable by ``type`` from the same case's YAML (that is why the script runs
first).

Every test restores the global plugin registries afterwards: a script registers
its node classes process-wide, and pytest runs the whole suite in one process.
Every case is copied into ``tmp_path`` first, because executing a script writes a
``__pycache__`` next to it and a checked-in case is never written to. No
OpenFOAM — a pipeline is only *built* here, never evaluated.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Iterator, cast

import pytest

from neofoam.algorithms.field_writer.write_control import WriteControl
from neofoam.core.plugin_system import PluginSystem
from neofoam.postprocess.config import PostProcessConfig, resolve_table
from neofoam.postprocess.script import load_script
from neofoam.postprocess.table import tables_for_case
from neofoam.tooling.workspace import CaseAccessError

CASES = Path(__file__).parent / "cases"


def _case(tmp_path: Path, name: str) -> Path:
    """A private copy of a checked-in case — executing its script writes bytecode."""
    case_dir = tmp_path / name
    shutil.copytree(CASES / name, case_dir)
    return case_dir


@pytest.fixture(autouse=True)
def restore_plugin_registries() -> Iterator[None]:
    """Unregister what a script registered, so the next test sees a clean union."""
    before = {family: list(plugins) for family, plugins in PluginSystem.list_plugins().items()}
    yield
    for family, plugins in PluginSystem.list_plugins().items():
        for plugin_cls in list(plugins):
            if plugin_cls not in before.get(family, []):
                PluginSystem.remove_plugin_model(family, plugin_cls)


def test_script_table_is_the_pipeline_the_spec_file_resolves_to(tmp_path: Path) -> None:
    tables = load_script(_case(tmp_path, "script_table"))
    declared = PostProcessConfig.load(case_dir=CASES / "one_table/system/postProcess.yaml")

    assert tables is not None
    assert tables.tables[0].pipeline == resolve_table(declared.tables[0]).pipeline


def test_table_name_is_the_decorated_file_stem(tmp_path: Path) -> None:
    tables = load_script(_case(tmp_path, "script_table"))
    assert tables is not None
    assert [table.name for table in tables.tables] == ["volume_p"]


def test_case_without_a_script_loads_nothing(tmp_path: Path) -> None:
    assert load_script(tmp_path) is None


def test_a_relative_path_leaving_the_case_is_refused(tmp_path: Path) -> None:
    shutil.copy(CASES / "script_table/system/postProcess.py", tmp_path / "outside.py")
    case_dir = tmp_path / "case"
    case_dir.mkdir()

    with pytest.raises(CaseAccessError):
        load_script(case_dir, "../outside.py")


def test_a_symlink_leaving_the_case_is_refused(tmp_path: Path) -> None:
    shutil.copy(CASES / "script_table/system/postProcess.py", tmp_path / "outside.py")
    case_dir = tmp_path / "case"
    (case_dir / "system").mkdir(parents=True)
    (case_dir / "system/postProcess.py").symlink_to(tmp_path / "outside.py")

    with pytest.raises(CaseAccessError):
        load_script(case_dir)


@pytest.mark.parametrize("case", ["no_table_set", "two_table_sets"])
def test_script_must_define_exactly_one_table_set(case: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly one module-level TableSet"):
        load_script(_case(tmp_path, case))


def test_a_node_the_script_registers_is_usable_by_type_in_the_same_case(tmp_path: Path) -> None:
    tables = tables_for_case(_case(tmp_path, "custom_node"))

    by_name = {table.name: table for table in tables.tables}
    assert set(by_name) == {"clipped_p_script", "clipped_p_declared"}
    declared, scripted = by_name["clipped_p_declared"], by_name["clipped_p_script"]
    assert type(declared.pipeline.steps[0]) is type(scripted.pipeline.steps[0])


def test_loading_the_same_script_twice_still_resolves_its_node(tmp_path: Path) -> None:
    case_dir = _case(tmp_path, "custom_node")
    first = tables_for_case(case_dir)
    second = tables_for_case(case_dir)

    assert [table.name for table in second.tables] == [table.name for table in first.tables]


def test_a_plugin_outside_the_node_families_survives_the_next_script(tmp_path: Path) -> None:
    load_script(_case(tmp_path, "foreign_plugin"))

    load_script(_case(tmp_path, "script_table"))

    policy = cast(Any, WriteControl).create(policy={"write_control_type": "never"}).policy
    assert policy.write_control_type == "never"


def test_duplicate_table_name_names_both_origins(tmp_path: Path) -> None:
    case_dir = _case(tmp_path, "script_table")
    shutil.copy(CASES / "one_table/system/postProcess.yaml", case_dir / "system/postProcess.yaml")

    with pytest.raises(ValueError, match=r"postProcess.py.*postProcess.yaml"):
        tables_for_case(case_dir)


def test_a_script_that_fails_leaves_its_node_type_free_for_the_next_case(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="this case's script is broken"):
        load_script(_case(tmp_path, "failing_script"))

    tables = tables_for_case(_case(tmp_path, "custom_node"))
    assert [table.name for table in tables.tables] == [
        "clipped_p_script",
        "clipped_p_declared",
    ]


def test_a_case_declaring_neither_front_door_has_no_tables(tmp_path: Path) -> None:
    assert tables_for_case(tmp_path).tables == []
