# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Spec for the pure-Python condition vote and its OpenFOAM groupID fold."""

# NOTE: no `from __future__ import annotations` — keep the package style.

import pytest

from neofoam.algorithms.solution_loop.conditions import (
    ConditionVote,
    fold_conditions,
)


@pytest.mark.parametrize(
    "votes, exp_satisfied, exp_action",
    [
        ([], False, "end"),
        ([ConditionVote(satisfied=True)], True, "end"),
        ([ConditionVote(satisfied=False)], False, "end"),
        (
            [ConditionVote(satisfied=True), ConditionVote(satisfied=False)],
            True,
            "end",
        ),
        (
            [
                ConditionVote(satisfied=True, action="end"),
                ConditionVote(satisfied=True, action="abort"),
            ],
            True,
            "abort",
        ),
        (
            [
                ConditionVote(satisfied=True, action="end"),
                ConditionVote(satisfied=False, action="abort"),
            ],
            True,
            "end",
        ),
    ],
)
def test_ungrouped_votes_fold_by_or_with_abort_precedence(
    votes: list[ConditionVote], exp_satisfied: bool, exp_action: str
) -> None:
    # Ungrouped votes are each an independent trigger (OR); a non-triggering abort
    # must not colour the folded action.
    result = fold_conditions(votes)
    assert result.satisfied is exp_satisfied
    assert result.action == exp_action


def test_group_triggers_only_when_all_members_satisfied() -> None:
    partial = fold_conditions(
        [
            ConditionVote(satisfied=True, group_id=1),
            ConditionVote(satisfied=False, group_id=1),
        ]
    )
    assert partial.satisfied is False

    full = fold_conditions(
        [
            ConditionVote(satisfied=True, group_id=1),
            ConditionVote(satisfied=True, group_id=1),
        ]
    )
    assert full.satisfied is True


def test_group_id_zero_is_an_and_group() -> None:
    # group_id 0 is a real group (not ungrouped), so a partial group does not trigger.
    result = fold_conditions(
        [
            ConditionVote(satisfied=True, group_id=0),
            ConditionVote(satisfied=False, group_id=0),
        ]
    )
    assert result.satisfied is False


def test_or_across_groups_one_full_one_partial() -> None:
    result = fold_conditions(
        [
            ConditionVote(satisfied=True, group_id=1),  # group 1 full -> triggers
            ConditionVote(satisfied=True, group_id=2),
            ConditionVote(satisfied=False, group_id=2),  # group 2 partial
        ]
    )
    assert result.satisfied is True


def test_abort_propagates_from_a_fully_satisfied_group() -> None:
    result = fold_conditions(
        [
            ConditionVote(satisfied=True, group_id=1, action="abort"),
            ConditionVote(satisfied=True, group_id=1, action="abort"),
        ]
    )
    assert result.satisfied is True
    assert result.action == "abort"


def test_generator_input_is_materialised() -> None:
    # A single-pass generator must fold the same as a list.
    result = fold_conditions(
        v
        for v in [
            ConditionVote(satisfied=False),
            ConditionVote(satisfied=True),
        ]
    )
    assert result.satisfied is True


@pytest.mark.parametrize(
    "votes, gid",
    [
        (
            [
                ConditionVote(satisfied=True, group_id=7, action="end"),
                ConditionVote(satisfied=True, group_id=7, action="abort"),
            ],
            7,
        ),
        (
            [
                ConditionVote(satisfied=True, group_id=3, action="end"),
                ConditionVote(satisfied=True, group_id=3, action="abort"),
                ConditionVote(satisfied=False, group_id=3, action="end"),
            ],
            3,
        ),
    ],
)
def test_group_with_conflicting_satisfied_actions_raises_naming_the_group(
    votes: list[ConditionVote], gid: int
) -> None:
    # Two satisfied members disagree on action; a third unsatisfied member does not
    # rescue the group from the conflict.
    with pytest.raises(ValueError, match=rf"group {gid}"):
        fold_conditions(votes)
