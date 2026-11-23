# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Place a contributed wrapping loop around operations of the resolved graph.

A model or boundary condition contributes a loop that repeats part of the time
step; its ``OperationMetadata.wraps`` (:class:`~neofoam.framework.types.Placement`)
says which part:

- no operations: the whole body of the outermost loop (the time loop), or of the
  innermost wrapping loop already placed there::

    time_loop                       time_loop
        start_step                      couple
        solve_T      Placement()            start_step
        update_rho   ---------->            solve_T
        write                               update_rho
                                            write

- named operations: those and every operation downstream of them in their scope;
  the loop takes the place of the last of them::

    time_loop                       time_loop
        start_step                      start_step
        solve_T      ("solve_T",)       couple
        update_rho   ----------->           solve_T
        write                               update_rho
                                        write

  A named loop inside the outermost one (e.g. PIMPLE's ``inner_loop``) is held whole.

Loops are placed in the order given, outer first; one that would have to hold a
loop placed before it raises. An operation marked ``first`` runs only in the first
pass of the wrapping loops around it, one marked ``final`` only in their final
pass; dependencies that would read stale data are rejected.

Use ``wraps`` for a loop contributed from outside the solver; a solver's own loop
belongs in its ``StepBuilder``. Placing a loop leaves the operations it is given
unchanged and returns new containers.

Example::

    OperationMetadata(op_name="couple", wraps=Placement(operations=("solve_T",)))
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Iterator

from neofoam.framework.operations import IterativeOp, Operation
from neofoam.framework.types import Placement

Path = tuple[int, ...]


def _downstream(ops: list[Operation], seed: Callable[[Operation], bool]) -> list[Operation]:
    """The operations of a resolved scope that ``seed`` picks and all downstream of them."""
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


def _wrap(ops: list[Operation], loop: Operation) -> list[Operation]:
    """The resolved ``ops`` with ``loop`` placed; ``ValueError`` without a single scope."""
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
                walk(op.sub_operations, _inside(op, around))
            elif op.operation_name:
                located[op.operation_name] = (op, around)

    walk(ops, ())
    return located


def _inside(loop: Operation, around: tuple[str, ...]) -> tuple[str, ...]:
    """The wrapping loops around the body of ``loop``."""
    return (*around, loop.operation_name or "loop") if _is_wrapping(loop) else around


def _upstream(
    name: str, op: Operation, located: dict[str, tuple[Operation, tuple[str, ...]]]
) -> list[str]:
    """The operations ``op`` depends on: its ``depends_on``, and those naming it ``before``."""
    return [
        *op.depends_on,
        *(other for other, (other_op, _) in located.items() if name in other_op.before),
    ]


def _check_passes(ops: list[Operation]) -> None:
    """Raise ``ValueError`` when a dependency inside a shared wrapping loop crosses passes."""
    # An op not marked final must not depend on a final one, and a first op only on a
    # first one: either would run before its dependency, or on stale data.
    pairs = list(_sharing_a_loop(_around(ops)))
    for name, op, dependency, dependency_op, loop in pairs:
        if dependency_op.metadata.final and not op.metadata.final:
            raise ValueError(
                f"{name} runs in every pass of the wrapping loop {loop}, but depends "
                f"on {dependency}, which runs only in its final pass (final); mark {name} "
                f"final too or place the loop so that {name} is outside it"
            )
    for name, op, dependency, dependency_op, loop in pairs:
        if op.metadata.first and not dependency_op.metadata.first:
            raise ValueError(
                f"{name} runs only in the first pass of the wrapping loop {loop} "
                f"(first), but depends on {dependency}, which runs in every pass of it; "
                f"mark {dependency} first too or place the loop around it"
            )


def _sharing_a_loop(
    located: dict[str, tuple[Operation, tuple[str, ...]]],
) -> Iterator[tuple[str, Operation, str, Operation, str]]:
    """Each dependency pair inside a wrapping loop around both, with the innermost such loop."""
    for name, (op, around) in located.items():
        for dependency in _upstream(name, op, located):
            if dependency not in located:
                continue
            dependency_op, dependency_around = located[dependency]
            loop = _innermost_shared(around, dependency_around)
            if loop is not None:
                yield name, op, dependency, dependency_op, loop


def _innermost_shared(around: tuple[str, ...], other: tuple[str, ...]) -> str | None:
    """The innermost of the loops ``around`` that is also in ``other``."""
    shared = [loop for loop in around if loop in other]
    return shared[-1] if shared else None


def _top(ops: list[Operation], loop: Operation) -> Path:
    """The path to the scope a loop without operations wraps whole."""
    path = _outermost_path(ops, loop)
    scope = _scope(ops, path)
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


def _outermost_path(ops: list[Operation], loop: Operation) -> Path:
    """The path to the body of the single outermost loop, or to the root without one."""
    outer = [index for index, op in enumerate(ops) if isinstance(op.func, IterativeOp)]
    if len(outer) > 1:
        raise ValueError(
            f"{loop.operation_name} wraps the body of the outermost loop, but the graph "
            "has several; name its operations"
        )
    return (outer[0],) if outer else ()


def _outermost_loops(located: dict[str, tuple[Path, Operation]]) -> set[str]:
    """The names of the loops in the top scope."""
    return {name for name, (path, op) in located.items() if not path and _is_loop(op)}


def _check_named(
    loop: Operation, names: list[str], located: dict[str, tuple[Path, Operation]]
) -> None:
    """Raise ``ValueError`` unless ``names`` are operations inside the outermost loop."""
    # A loop of the top scope (the time loop) cannot be repeated by one inside it.
    outermost = _outermost_loops(located)
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


def _selected(
    ops: list[Operation], loop: Operation, placement: Placement
) -> tuple[Path, list[Operation]]:
    """The scope of the operations ``placement`` names, and what the loop takes there."""
    names = list(placement.operations or ())
    located = _located(ops, ())
    _check_named(loop, names, located)
    paths = {located[name][0] for name in names}
    if len(paths) > 1:
        raise ValueError(
            f"the operations {names} lie in more than one loop scope; "
            f"{loop.operation_name} wraps operations of one scope"
        )
    (path,) = paths
    scope = _scope(ops, path)
    seeds = {id(located[name][1]) for name in names}
    picked = _downstream(scope, lambda op: id(op) in seeds)
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
    """Every named operation in ``ops`` and below, with its scope's loop indices from the root."""
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
    """``scope`` with the ``selected`` operations moved into ``loop``, at the last one's place."""
    member_ids = {id(op) for op in selected}
    last = max(index for index, op in enumerate(scope) if id(op) in member_ids)
    # ``selected`` holds all downstream of its members: nothing earlier depends on it.
    earlier = [op for op in scope[:last] if id(op) not in member_ids]
    return [*earlier, _filled(loop, selected), *scope[last + 1 :]]
