# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Spec for actions: ``actions:`` entries in system/postProcess.yaml.

NeoFOAM ships no action type, so this module registers its own: ``recorder``
appends what it was called with to :data:`CALLS` (a module global, because the
runner executes a private copy of every entry). The model is driven as in
``test_model.py`` — LOAD a checked-in case, BUILD, call the operation with a
real Context and LoopState — so no OpenFOAM and no renderer are needed.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Literal, cast

import pytest

from neofoam.algorithms.solution_loop.loop_state import LoopState
from neofoam.framework.context import Context
from neofoam.postprocess.actions import Action, ActionRunner
from neofoam.postprocess.model import actions, build, postProcess
from neofoam.postprocess.table import tables_for_case

CASES = Path(__file__).parent / "cases"

#: ``(entry, label, step index)`` per ``recorder`` call; ``(entry, "finalize", -1)`` at the end.
CALLS: list[tuple[str, str, int]] = []


@Action.register
class Recorder(Action):
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


def _ctx(runner: ActionRunner, index: int) -> Context:
    return Context(
        fields={},
        models={"actions": runner},
        time=LoopState(value=0.1 * index, delta_t=0.1, end_time=1.0, index=index),
    )


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("cadence", ["post_process"]),
        ("actions", ["post_process", "actions"]),
    ],
    ids=["no_entries", "declared_entries"],
)
def test_the_actions_operation_follows_post_process_only_when_declared(
    tmp_path: Path, case: str, expected: list[str]
) -> None:
    runtime = postProcess.instantiate(case_dir=_case(tmp_path, case))

    operations = runtime.operations

    assert [op.operation_name for op in operations] == expected


def test_a_case_without_actions_builds_no_action_runner(tmp_path: Path) -> None:
    steps = build(tables_for_case(_case(tmp_path, "cadence")))

    assert [step.name for step in steps] == ["models.post_processor"]


def test_declared_actions_build_the_action_runner(tmp_path: Path) -> None:
    steps = build(tables_for_case(_case(tmp_path, "actions")))

    assert [step.name for step in steps] == ["models.post_processor", "models.actions"]
    assert isinstance(steps[1].initializer({}), ActionRunner)


def test_each_action_runs_on_its_own_write_steps(tmp_path: Path) -> None:
    steps = build(tables_for_case(_case(tmp_path, "actions")))
    runner = cast(ActionRunner, steps[1].initializer({}))

    for index in range(3):
        actions(None, _ctx(runner, index))

    assert CALLS == [
        ("every_step_view", "", 0),
        ("every_second_step_view", "coarse", 0),
        ("every_step_view", "", 1),
        ("every_step_view", "", 2),
        ("every_second_step_view", "coarse", 2),
    ]


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("unknown_action", r"action 'view'.*registered action types: \[.*'recorder'"),
        ("duplicate_action", "duplicate action entries"),
    ],
    ids=["unknown_type", "duplicate_name"],
)
def test_an_invalid_action_declaration_is_refused(tmp_path: Path, case: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        tables_for_case(_case(tmp_path, case))


def test_the_runner_finalizes_every_entry(tmp_path: Path) -> None:
    steps = build(tables_for_case(_case(tmp_path, "actions")))
    runner = cast(ActionRunner, steps[1].initializer({}))

    runner.finalize()

    assert CALLS == [
        ("every_step_view", "finalize", -1),
        ("every_second_step_view", "finalize", -1),
    ]
