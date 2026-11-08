# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""A loop that wraps fields: the resolver nests their solution inside it.

A model or boundary condition contributes a loop operation that declares the
fields it repeats (``OperationMetadata.wraps``), not the operations: it does not
know the solver's names. :func:`wrap` moves the operations that solve those
fields, and every operation downstream of them, into the loop, which takes the
place of the last of them::

    time_loop                       time_loop
        start_step                      start_step
        solve_T          wraps T        left.couple
        update_rho       ------->           solve_T
        write                               update_rho
                                        write

Downstream means reachable over ``depends_on`` / ``before`` within the scope (the
time-step body or a loop body) that holds the solving operations.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

from neofoam.framework.operations import IterativeOp, Operation

# The path to a scope: the index of each loop on the way down from the root.
Path = tuple[int, ...]


def downstream(ops: list[Operation], seed: Callable[[Operation], bool]) -> list[Operation]:
    """The operations of one resolved scope that ``seed`` picks and every one downstream of them.

    The result keeps the order of ``ops``.
    """
    selected: list[Operation] = []
    names: set[str] = set()
    before_targets: set[str] = set()
    # A resolved scope is in dependency order, so one forward pass sees every
    # operation after the ones it depends on.
    for op in ops:
        if not (
            seed(op) or names.intersection(op.depends_on) or op.operation_name in before_targets
        ):
            continue
        selected.append(op)
        if op.operation_name:
            names.add(op.operation_name)
        before_targets.update(op.before)
    return selected


def wrap(ops: list[Operation], loop: Operation) -> list[Operation]:
    """The resolved ``ops`` with the fields ``loop.metadata.wraps`` names solved inside ``loop``.

    The loop takes the place of the last selected operation; the other operations
    of its scope keep their order. It solves what its body solves, so a second
    loop that wraps one of the fields wraps the first.

    Raises ``ValueError`` when no operation solves one of the fields, or when the
    selected operations lie in more than one scope.
    """
    fields = sorted(loop.metadata.wraps)
    found = _found(ops, lambda op: bool(set(fields).intersection(op.metadata.solves)), ())
    solved = {field for _path, picked in found for op in picked for field in op.metadata.solves}
    missing = [field for field in fields if field not in solved]
    if missing:
        raise ValueError(
            f"{loop.operation_name} wraps {missing}, which no operation solves; it needs "
            "an operation declared with solves=[...] for each field"
        )
    if len(found) > 1:
        raise ValueError(
            f"the operations solving {fields} lie in more than one loop scope; "
            f"{loop.operation_name} wraps operations of one scope"
        )
    ((path, picked),) = found
    return _in_scope(ops, path, lambda scope: _wrapped(scope, picked, loop))


def _found(
    ops: list[Operation], seed: Callable[[Operation], bool], path: Path
) -> list[tuple[Path, list[Operation]]]:
    """Every scope where ``seed`` picks operations, and what it selects there.

    A selected loop is taken whole: the scopes inside it are not searched.
    """
    picked = downstream(ops, seed)
    found = [(path, picked)] if picked else []
    taken = {id(op) for op in picked}
    for index, op in enumerate(ops):
        if isinstance(op.func, IterativeOp) and id(op) not in taken:
            found += _found(op.sub_operations, seed, (*path, index))
    return found


def _in_scope(
    ops: list[Operation], path: Path, change: Callable[[list[Operation]], list[Operation]]
) -> list[Operation]:
    """``ops`` with the scope at ``path`` changed by ``change``."""
    if not path:
        return change(ops)
    index, rest = path[0], path[1:]
    loop = dataclasses.replace(
        ops[index], sub_operations=_in_scope(ops[index].sub_operations, rest, change)
    )
    return [*ops[:index], loop, *ops[index + 1 :]]


def _wrapped(scope: list[Operation], selected: list[Operation], loop: Operation) -> list[Operation]:
    """``scope`` with the ``selected`` operations moved into ``loop``, at the last one's place."""
    member_ids = {id(op) for op in selected}
    last = max(index for index, op in enumerate(scope) if id(op) in member_ids)
    solves = sorted({field for op in selected for field in op.metadata.solves})
    filled = dataclasses.replace(
        loop,
        metadata=dataclasses.replace(loop.metadata, solves=solves),
        sub_operations=list(selected),
    )
    # Operations before the last member that are not members do not depend on one
    # (they would be downstream), so running them before the loop keeps every edge.
    earlier = [op for op in scope[:last] if id(op) not in member_ids]
    return [*earlier, filled, *scope[last + 1 :]]
