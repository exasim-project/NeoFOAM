# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for the solution-loop model-owned gather-point interfaces."""

# NOTE: no `from __future__ import annotations` — keep annotations live.

import ast
from pathlib import Path

import pytest

import neofoam.algorithms.solution_loop as loop_pkg
from neofoam.algorithms.solution_loop.conditions import ConditionVote
from neofoam.algorithms.solution_loop.interfaces import (
    VGREAT,
    loopCondition,
    solutionLoop,
    timeStepConstraint,
)
from neofoam.framework.model import ModelInterface


def test_loop_interfaces_are_owned_by_the_loop_model() -> None:
    assert isinstance(timeStepConstraint, ModelInterface)
    assert isinstance(loopCondition, ModelInterface)
    assert timeStepConstraint.owner is solutionLoop
    assert loopCondition.owner is solutionLoop
    assert solutionLoop.declared_interfaces["timeStepConstraint"] is timeStepConstraint
    assert solutionLoop.declared_interfaces["loopCondition"] is loopCondition


@pytest.mark.parametrize(
    "limits, expected",
    [([], VGREAT), ([2.0, 1.0, 3.0], 1.0), ([5.0], 5.0)],
)
def test_time_step_constraint_folds_with_min(
    limits: list[float], expected: float
) -> None:
    assert timeStepConstraint.fold(limits) == expected


def test_loop_condition_folds_empty_to_not_satisfied() -> None:
    assert loopCondition.fold([]).satisfied is False


@pytest.mark.parametrize(
    "votes, expected",
    [
        ([ConditionVote(satisfied=True)], True),
        ([ConditionVote(satisfied=False)], False),
        ([ConditionVote(satisfied=False), ConditionVote(satisfied=True)], True),
    ],
)
def test_loop_condition_folds_votes(votes: list[ConditionVote], expected: bool) -> None:
    assert loopCondition.fold(votes).satisfied is expected


def _imported_names(source: str) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_framework_solution_loop_imports_no_pybfoam() -> None:
    pkg_dir = Path(loop_pkg.__file__).parent
    offenders = {
        path.name: imported
        for path in pkg_dir.glob("*.py")
        if "pybFoam" in (imported := _imported_names(path.read_text()))
    }
    assert not offenders, offenders
