# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Spec for the script front door: executing a case's system/setFields.py.

The scripts are checked-in case files (``cases/*/system/setFields.py``) — exactly
what a user writes — because executing them *is* the behaviour under test. Two
properties carry the design: a script builds its regions from the *post-processing*
selectors (that is why ``assign`` takes a ``Selector`` and nothing else), and a
selector the script registers is selectable by ``type`` from the same case's YAML
(that is why the script runs first).

Every test restores the global plugin registries afterwards: a script registers
its selector classes process-wide, and pytest runs the whole suite in one process.
Every case is copied into ``tmp_path`` first, because executing a script writes a
``__pycache__`` next to it and a checked-in case is never written to. No OpenFOAM —
a declaration is only *built* here, never applied.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterator

import pytest

from neofoam.core.plugin_system import PluginSystem
from neofoam.postprocess import Binary, Box, Sphere
from neofoam.preprocess.script import load_script, set_fields_for_case
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


def test_a_script_declares_its_defaults_and_regions(tmp_path: Path) -> None:
    setup = load_script(_case(tmp_path, "script_regions"))

    assert setup is not None
    assert setup.defaults == {"alpha.water": 0.0}
    assert setup.regions == [
        (
            Binary(
                op="or",
                left=Box(min=(0, 0, -1), max=(0.1461, 0.292, 1)),
                right=Sphere(center=(0, 0, 0), radius=0.25),
            ),
            {"alpha.water": 1.0},
        )
    ]


def test_case_without_a_script_loads_nothing(tmp_path: Path) -> None:
    assert load_script(tmp_path) is None


@pytest.mark.parametrize("case", ["no_set_fields", "two_set_fields"])
def test_script_must_define_exactly_one_set_fields(case: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly one module-level SetFields"):
        load_script(_case(tmp_path, case))


def test_a_relative_path_leaving_the_case_is_refused(tmp_path: Path) -> None:
    shutil.copy(CASES / "script_regions/system/setFields.py", tmp_path / "outside.py")
    case_dir = tmp_path / "case"
    case_dir.mkdir()

    with pytest.raises(CaseAccessError):
        load_script(case_dir, "../outside.py")


def test_a_symlink_leaving_the_case_is_refused(tmp_path: Path) -> None:
    shutil.copy(CASES / "script_regions/system/setFields.py", tmp_path / "outside.py")
    case_dir = tmp_path / "case"
    (case_dir / "system").mkdir(parents=True)
    (case_dir / "system/setFields.py").symlink_to(tmp_path / "outside.py")

    with pytest.raises(CaseAccessError):
        load_script(case_dir)


def test_a_selector_the_script_registers_is_usable_by_type_in_the_same_case(
    tmp_path: Path,
) -> None:
    setup = set_fields_for_case(_case(tmp_path, "custom_selector"))

    selector, values = setup.regions[0]
    assert type(selector).__name__ == "LeftHalf"
    assert values == {"alpha.water": 1.0}


def test_loading_the_same_case_twice_still_resolves_its_selector(tmp_path: Path) -> None:
    case_dir = _case(tmp_path, "custom_selector")
    first = set_fields_for_case(case_dir)

    second = set_fields_for_case(case_dir)

    assert [type(s).__name__ for s, _ in second.regions] == [
        type(s).__name__ for s, _ in first.regions
    ]


def test_the_spec_files_regions_come_after_the_scripts(tmp_path: Path) -> None:
    case_dir = _case(tmp_path, "script_regions")
    shutil.copy(CASES / "declared/system/setFields.yaml", case_dir / "system/setFields.yaml")

    setup = set_fields_for_case(case_dir)

    assert [type(selector).__name__ for selector, _ in setup.regions] == [
        "Binary",  # the script's
        "Box",  # the spec file's first
        "Binary",  # the spec file's second
    ]


def test_a_case_declaring_only_a_spec_file_needs_no_script(tmp_path: Path) -> None:
    setup = set_fields_for_case(_case(tmp_path, "declared"))

    assert setup.defaults == {"alpha.water": 0.0}
    assert len(setup.regions) == 2


def test_a_case_declaring_only_a_script_needs_no_spec_file(tmp_path: Path) -> None:
    setup = set_fields_for_case(_case(tmp_path, "script_regions"))

    assert len(setup.regions) == 1


def test_a_case_declaring_neither_front_door_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="declares neither system/setFields.py"):
        set_fields_for_case(tmp_path)
