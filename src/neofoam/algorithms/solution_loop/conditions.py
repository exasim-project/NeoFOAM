# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Loop-condition votes and their OpenFOAM ``runTimeControl`` groupID fold.

A :class:`ConditionVote` is one stop criterion's verdict: ``satisfied`` (the
OpenFOAM "condition satisfied" = this stop criterion is met), a ``group_id``
(OpenFOAM ``groupID``; ``-1`` = ungrouped, an independent trigger) and an
``action`` (``satisfiedAction``: a clean ``end`` vs a failure ``abort``).

:func:`fold_conditions` combines votes the way ``functionObjects::runTimeControl``
does: **AND within a group** (a ``group_id >= 0`` group triggers only when every
member is satisfied), **OR across** groups and ungrouped votes (the run stops when
any trigger fires). An empty input is *not satisfied* (keep running). The folded
``action`` is ``abort`` iff any *triggering* member is ``abort``. A group whose
satisfied members disagree on ``action`` is a configuration error and raises
``ValueError``.

Leaf module: imports only the stdlib so ``algorithms.solution_loop`` stays
pybFoam-free (field reads live solver-side).
"""

from dataclasses import dataclass
from typing import Iterable, Literal

Action = Literal["end", "abort"]


@dataclass(frozen=True)
class ConditionVote:
    """One stop criterion's verdict (see module docstring)."""

    satisfied: bool
    group_id: int = -1
    action: Action = "end"


def fold_conditions(votes: Iterable[ConditionVote]) -> ConditionVote:
    """Combine *votes* by OpenFOAM ``runTimeControl`` groupID semantics.

    stop = OR across groups; ``group_id < 0`` -> each vote independent,
    ``group_id >= 0`` -> AND within the group. Empty -> ``satisfied=False``.
    The folded ``action`` is ``abort`` iff a triggering member aborts, else
    ``end``.

    Raises:
        ValueError: a ``group_id >= 0`` group whose *satisfied* members disagree
            on ``action`` (naming the group).
    """
    votes = list(votes)
    groups: dict[int, list[ConditionVote]] = {}
    ungrouped: list[ConditionVote] = []
    for vote in votes:
        if vote.group_id < 0:
            ungrouped.append(vote)
        else:
            groups.setdefault(vote.group_id, []).append(vote)

    triggering_actions: list[Action] = []

    # Ungrouped: each vote is its own independent trigger.
    for vote in ungrouped:
        if vote.satisfied:
            triggering_actions.append(vote.action)

    # Grouped: AND within the group; satisfied members must agree on the action.
    for group_id, members in groups.items():
        satisfied = [member for member in members if member.satisfied]
        actions = {member.action for member in satisfied}
        if len(actions) > 1:
            raise ValueError(
                f"condition group {group_id}: satisfied members disagree on "
                f"action {sorted(actions)} (a group must agree on end vs abort)."
            )
        if satisfied and len(satisfied) == len(members):
            triggering_actions.append(next(iter(actions)))

    if not triggering_actions:
        return ConditionVote(satisfied=False)
    action: Action = "abort" if "abort" in triggering_actions else "end"
    return ConditionVote(satisfied=True, action=action)
