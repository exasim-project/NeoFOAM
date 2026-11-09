# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""A wrapping loop: the resolver places it around operations of the graph.

A model or boundary condition contributes a loop operation that repeats part of
the solver's time step. Its ``OperationMetadata.wraps``
(:class:`~neofoam.framework.types.Placement`) says which part:

- no operations: the whole body of the top scope, the body of the outermost loop
  (the time loop). Once a wrapping loop sits there, the next one wraps the whole
  body of the innermost of them::

    time_loop                       time_loop
        start_step                      couple
        solve_T      Placement()            start_step
        update_rho   ---------->            solve_T
        write                               update_rho
                                            write

- named operations: those and, with ``downstream``, every operation downstream of
  them over ``depends_on`` / ``before`` within their scope (the time-step body or
  a loop body). The loop takes the place of the last of them::

    time_loop                       time_loop
        start_step                      start_step
        solve_T      ("solve_T",)       couple
        update_rho   ----------->           solve_T
        write                               update_rho
                                        write

  A name may be a loop inside the outermost one (e.g. a PIMPLE ``inner_loop``):
  the wrapping loop holds it whole, as one operation, and downstream of it is
  downstream of anything inside it::

    time_loop                       time_loop
        set_time_step                   set_time_step
        increment_time                  increment_time
        inner_loop   ("inner_loop",)    couple
            solve_U  -------------->        inner_loop
            solve_p                             solve_U
        write                                   solve_p
                                            write

Loops are placed in the order given, outer first: a loop whose operations lie
inside one placed before nests inside it; one that would have to hold a loop
placed before raises.

An operation marked ``first`` (``OperationMetadata.first``, e.g. the start of a
time step) runs only in the first pass of the wrapping loops around it; it may
not depend on an operation that runs in every pass of one of them
(:func:`check_first`). One marked ``final`` (``OperationMetadata.final``, e.g.
writing the results) runs only in their final pass; no operation that runs in
an other pass of one of them may depend on it (:func:`check_final`).
"""

from __future__ import annotations

import dataclasses
from typing import Callable

from neofoam.framework.operations import IterativeOp, Operation
from neofoam.framework.types import Placement

# The path to a scope: the index of each loop on the way down from the root.
Path = tuple[int, ...]


def downstream(ops: list[Operation], seed: Callable[[Operation], bool]) -> list[Operation]:
    """The operations of one resolved scope that ``seed`` picks and every one downstream of them.

    A loop counts with the edges and names of everything inside it. The result
    keeps the order of ``ops``.
    """
    selected: list[Operation] = []
    names: set[str] = set()
    before_targets: set[str] = set()
    # A resolved scope is in dependency order, so one forward pass sees every
    # operation after the ones it depends on.
    for op in ops:
        depends_on, before, own = _edges(op)
        if not (seed(op) or names.intersection(depends_on) or before_targets.intersection(own)):
            continue
        selected.append(op)
        names |= own
        before_targets.update(before)
    return selected


def wrap(ops: list[Operation], loop: Operation) -> list[Operation]:
    """The resolved ``ops`` with ``loop`` placed as ``loop.metadata.wraps`` says.

    Raises ``ValueError`` when an operation name is unknown or names the
    outermost loop, when the named operations lie in more than one scope, when
    the loop would have to hold a wrapping loop placed before it, or when no top
    scope can be told.
    """
    placement = loop.metadata.wraps
    assert placement is not None
    if placement.operations is None:
        path = _top(ops, loop)
        return _in_scope(ops, path, lambda scope: [_filled(loop, scope)])
    path, picked = _selected(ops, loop, placement)
    return _in_scope(ops, path, lambda scope: _wrapped(scope, picked, loop))


def _around(ops: list[Operation]) -> dict[str, tuple[Operation, tuple[str, ...]]]:
    """Every named operation of ``ops`` with the wrapping loops around it, outer first."""
    located: dict[str, tuple[Operation, tuple[str, ...]]] = {}

    def walk(scope: list[Operation], around: tuple[str, ...]) -> None:
        for op in scope:
            if _is_loop(op):
                inner = (*around, op.operation_name or "loop") if _is_wrapping(op) else around
                walk(op.sub_operations, inner)
            elif op.operation_name:
                located[op.operation_name] = (op, around)

    walk(ops, ())
    return located


def _upstream(
    name: str, op: Operation, located: dict[str, tuple[Operation, tuple[str, ...]]]
) -> list[str]:
    """The operations ``op`` depends on: its ``depends_on``, and those naming it ``before``."""
    return [*op.depends_on, *(other for other, (o, _) in located.items() if name in o.before)]


def check_final(ops: list[Operation]) -> None:
    """Raise ``ValueError`` when an operation not marked ``final`` depends on one that is.

    An operation marked ``final`` (``OperationMetadata.final``) runs only in the
    final pass of the wrapping loops around it; one that depends on it (its
    ``depends_on``, or the final one's ``before``) and runs in the other passes of
    a wrapping loop around both would run before it, or on stale data.
    """
    located = _around(ops)
    for name, (op, around) in located.items():
        if op.metadata.final or not around:
            continue
        for dependency in _upstream(name, op, located):
            if dependency not in located:
                continue
            dependency_op, dependency_around = located[dependency]
            shared = [loop for loop in around if loop in dependency_around]
            if shared and dependency_op.metadata.final:
                raise ValueError(
                    f"{name} runs in every pass of the wrapping loop {shared[-1]}, but depends "
                    f"on {dependency}, which runs only in its final pass (final); mark {name} "
                    f"final too or place the loop so that {name} is outside it"
                )


def check_first(ops: list[Operation]) -> None:
    """Raise ``ValueError`` when an operation marked ``first`` depends on one that is not.

    An operation marked ``first`` (``OperationMetadata.first``) runs only in the
    first pass of the wrapping loops around it; one it depends on (its
    ``depends_on``, or the other's ``before``) that runs in every pass of a
    wrapping loop around both would leave it on stale data.
    """
    located = _around(ops)
    for name, (op, around) in located.items():
        if not op.metadata.first or not around:
            continue
        for dependency in _upstream(name, op, located):
            if dependency not in located:
                continue
            dependency_op, dependency_around = located[dependency]
            shared = [loop for loop in around if loop in dependency_around]
            if shared and not dependency_op.metadata.first:
                raise ValueError(
                    f"{name} runs only in the first pass of the wrapping loop {shared[-1]} "
                    f"(first), but depends on {dependency}, which runs in every pass of it; "
                    f"mark {dependency} first too or place the loop around it"
                )


def _top(ops: list[Operation], loop: Operation) -> Path:
    """The path to the scope a loop without operations wraps whole."""
    path: Path = ()
    scope = ops
    outer = [index for index, op in enumerate(scope) if isinstance(op.func, IterativeOp)]
    if len(outer) > 1:
        raise ValueError(
            f"{loop.operation_name} wraps the body of the outermost loop, but the graph "
            "has several; name its operations"
        )
    if outer:
        path = (outer[0],)
        scope = scope[outer[0]].sub_operations
    while True:
        wrapping = [index for index, op in enumerate(scope) if _is_wrapping(op)]
        if not wrapping:
            return path
        if len(wrapping) > 1:
            raise ValueError(
                f"{loop.operation_name} wraps the body of the innermost wrapping loop, "
                "but a scope holds several; name its operations"
            )
        path = (*path, wrapping[0])
        scope = scope[wrapping[0]].sub_operations


def _selected(
    ops: list[Operation], loop: Operation, placement: Placement
) -> tuple[Path, list[Operation]]:
    """The scope of the operations ``placement`` names, and what the loop takes there."""
    names = list(placement.operations or ())
    located = _located(ops, ())
    # A loop of the top scope (the time loop) cannot be repeated by one inside it.
    outermost = {name for name, (path, op) in located.items() if not path and _is_loop(op)}
    unknown = [name for name in names if name not in located]
    if unknown:
        available = sorted(name for name in located if name not in outermost)
        raise ValueError(
            f"{loop.operation_name} wraps {unknown}, which name no operation; "
            f"the operations are {available}"
        )
    loops = [name for name in names if name in outermost]
    if loops:
        raise ValueError(
            f"{loop.operation_name} wraps {loops}, the outermost loop; it wraps operations "
            "or loops inside it"
        )
    paths = {located[name][0] for name in names}
    if len(paths) > 1:
        raise ValueError(
            f"the operations {names} lie in more than one loop scope; "
            f"{loop.operation_name} wraps operations of one scope"
        )
    (path,) = paths
    scope = _scope(ops, path)
    seeds = {id(located[name][1]) for name in names}
    if placement.downstream:
        picked = downstream(scope, lambda op: id(op) in seeds)
    else:
        picked = [op for op in scope if id(op) in seeds]
    held = _wrapping_in(picked)
    if held:
        raise ValueError(
            f"{loop.operation_name} would hold the coupling loop {held} placed before it; "
            "list the outer coupling first"
        )
    return path, picked


def _wrapping_in(ops: list[Operation]) -> list[str | None]:
    """The names of the wrapping loops among ``ops`` and inside their loops."""
    held: list[str | None] = []
    for op in ops:
        if _is_wrapping(op):
            held.append(op.operation_name)
        elif _is_loop(op):
            held.extend(_wrapping_in(op.sub_operations))
    return held


def _located(ops: list[Operation], path: Path) -> dict[str, tuple[Path, Operation]]:
    """Every named operation in ``ops`` and below, with the path to its scope."""
    located: dict[str, tuple[Path, Operation]] = {}
    for index, op in enumerate(ops):
        if op.operation_name:
            located[op.operation_name] = (path, op)
        if _is_loop(op):
            located.update(_located(op.sub_operations, (*path, index)))
    return located


def _edges(op: Operation) -> tuple[set[str], set[str], set[str]]:
    """``depends_on``, ``before`` and names of ``op`` and everything inside it."""
    depends_on, before = set(op.depends_on), set(op.before)
    own = {op.operation_name} if op.operation_name else set()
    if _is_loop(op):
        for inner in op.sub_operations:
            inner_depends, inner_before, inner_own = _edges(inner)
            depends_on |= inner_depends
            before |= inner_before
            own |= inner_own
    return depends_on - own, before - own, own


def _is_loop(op: Operation) -> bool:
    return isinstance(op.func, IterativeOp)


def _is_wrapping(op: Operation) -> bool:
    return _is_loop(op) and op.metadata.wraps is not None


def _scope(ops: list[Operation], path: Path) -> list[Operation]:
    for index in path:
        ops = ops[index].sub_operations
    return ops


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


def _filled(loop: Operation, body: list[Operation]) -> Operation:
    return dataclasses.replace(loop, sub_operations=list(body))


def _wrapped(scope: list[Operation], selected: list[Operation], loop: Operation) -> list[Operation]:
    """``scope`` with the ``selected`` operations moved into ``loop``, at the last one's place.

    Raises ``ValueError`` when an operation left before the loop depends on one in it.
    """
    member_ids = {id(op) for op in selected}
    last = max(index for index, op in enumerate(scope) if id(op) in member_ids)
    earlier = [op for op in scope[:last] if id(op) not in member_ids]
    # Without downstream an operation between the members may depend on one of
    # them; running it before the loop would break that edge.
    after = {id(op) for op in downstream(scope[:last], lambda op: id(op) in member_ids)}
    broken = [op.operation_name for op in earlier if id(op) in after]
    if broken:
        raise ValueError(
            f"{broken} depend on operations {loop.operation_name} wraps but run before "
            "it; wrap them too"
        )
    return [*earlier, _filled(loop, selected), *scope[last + 1 :]]
