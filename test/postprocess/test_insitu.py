# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Spec for the in-situ hook: ``insitu:`` entries in system/postProcess.yaml.

NeoFOAM ships no in-situ type, so this module registers its own: ``recorder``
appends what it was called with to :data:`CALLS` (a module global, because the
runner executes a private copy of every entry). The model is driven as in
``test_model.py`` — LOAD a checked-in case, BUILD, call the operation with a
real Context and LoopState — so no OpenFOAM and no renderer are needed.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, cast

import pytest

from neofoam.algorithms.solution_loop.loop_state import LoopState
from neofoam.framework.context import Context
from neofoam.postprocess.insitu import InSitu, InSituRunner
from neofoam.postprocess.model import build, insitu, postProcess
from neofoam.postprocess.table import TableSet, tables_for_case

CASES = Path(__file__).parent / "cases"

#: ``(entry, label, step index)`` per ``recorder`` call; ``(entry, "finalize", -1)`` at the end.
CALLS: list[tuple[str, str, int]] = []


@InSitu.register
class Recorder(InSitu):
    type: Literal["recorder"] = "recorder"
    label: str = ""

    def execute(self, ctx: Context) -> None:
        CALLS.append((self.name, self.label, ctx.time.index))

    def finalize(self) -> None:
        CALLS.append((self.name, "finalize", -1))


@pytest.fixture(autouse=True)
def _no_calls() -> None:
    CALLS.clear()


def _case(tmp_path: Path, name: str) -> Path:
    case_dir = tmp_path / name
    shutil.copytree(CASES / name, case_dir)
    return case_dir


def _ctx(runner: InSituRunner, index: int) -> Context:
    return Context(
        fields={},
        models={"insitu": runner},
        time=LoopState(value=0.1 * index, delta_t=0.1, end_time=1.0, index=index),
    )


def _op_names(tables: TableSet) -> list[str]:
    return [
        op.operation_name
        for op in postProcess._build_operations_for(SimpleNamespace(config=tables))
    ]


def test_a_case_without_insitu_entries_has_no_insitu_operation(tmp_path: Path) -> None:
    tables = tables_for_case(_case(tmp_path, "cadence"))

    assert _op_names(tables) == ["post_process"]


def test_a_case_without_insitu_entries_builds_no_insitu_runner(tmp_path: Path) -> None:
    steps = build(tables_for_case(_case(tmp_path, "cadence")))

    assert [step.name for step in steps] == ["models.post_processor"]


def test_declared_insitu_entries_add_the_insitu_operation_after_post_process(
    tmp_path: Path,
) -> None:
    tables = tables_for_case(_case(tmp_path, "insitu"))

    assert _op_names(tables) == ["post_process", "insitu"]


def test_declared_insitu_entries_build_the_insitu_runner(tmp_path: Path) -> None:
    steps = build(tables_for_case(_case(tmp_path, "insitu")))

    assert [step.name for step in steps] == ["models.post_processor", "models.insitu"]
    assert isinstance(steps[1].initializer({}), InSituRunner)


def test_each_insitu_entry_runs_on_its_own_write_steps(tmp_path: Path) -> None:
    steps = build(tables_for_case(_case(tmp_path, "insitu")))
    runner = cast(InSituRunner, steps[1].initializer({}))

    for index in range(3):
        insitu(None, _ctx(runner, index))

    assert CALLS == [
        ("every_step_view", "", 0),
        ("every_second_step_view", "coarse", 0),
        ("every_step_view", "", 1),
        ("every_step_view", "", 2),
        ("every_second_step_view", "coarse", 2),
    ]


def test_an_unknown_insitu_type_lists_the_registered_ones(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"insitu 'view'.*registered insitu types: \[.*'recorder'"):
        tables_for_case(_case(tmp_path, "unknown_insitu"))


def test_two_insitu_entries_sharing_a_name_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate insitu entries"):
        tables_for_case(_case(tmp_path, "duplicate_insitu"))


def test_the_runner_finalizes_every_entry(tmp_path: Path) -> None:
    steps = build(tables_for_case(_case(tmp_path, "insitu")))
    runner = cast(InSituRunner, steps[1].initializer({}))

    runner.finalize()

    assert CALLS == [
        ("every_step_view", "finalize", -1),
        ("every_second_step_view", "finalize", -1),
    ]
