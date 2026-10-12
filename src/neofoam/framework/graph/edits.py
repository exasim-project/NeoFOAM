# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Changes to a solver's graph registered at runtime, by a boundary condition or model.

A solver declares which fields its operations solve (``solves=[...]``); it does
not know which boundary conditions or models will need its graph changed. Those
register their change on ``ctx.graph`` (a :class:`GraphEdits`) while the case is
initialized, and the runtime applies them to the resolved graph::

    ctx = runtime.initialize()
    builder, model_ops = runtime.execution_graph(ctx=ctx)
    graph = ctx.graph.apply(DAGResolver().resolve(builder, model_ops))

Without a registered edit, :meth:`GraphEdits.apply` returns the graph it is given.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from neofoam.framework.coupling.loop import CoupledSide, CouplingOp
from neofoam.framework.operations import IterativeOp, Operation, StepBuilder
from neofoam.framework.types import OperationMetadata


@dataclass(frozen=True)
class Selector:
    """The operations that solve ``field`` and every operation downstream of them.

    Use :meth:`GraphEdits.solving` to get one. It is lazy: it picks operations
    only when the edit is applied, from the scope (time-step body or loop body)
    that holds the solving operations. Downstream means reachable over
    ``depends_on`` / ``before`` within that scope. A loop built by a graph edit
    solves what its body solves, so a second edit on the same field wraps it.

    Example::

        ctx.graph.solving("T")
    """

    field: str

    def select(self, ops: list[Operation]) -> list[Operation]:
        """The operations of one resolved scope that this selector picks, in order."""
        selected: list[Operation] = []
        names: set[str] = set()
        before_targets: set[str] = set()
        # A resolved scope is in dependency order, so one forward pass sees every
        # operation after the ones it depends on.
        for op in ops:
            if not (
                self.field in op.metadata.solves
                or names.intersection(op.depends_on)
                or op.operation_name in before_targets
            ):
                continue
            selected.append(op)
            if op.operation_name:
                names.add(op.operation_name)
            before_targets.update(op.before)
        return selected


@dataclass(frozen=True)
class _Wrap:
    """One registered wrap: the operations it selects, the loop around them, its coupled side."""

    ops: Selector
    with_: IterativeOp
    side: CoupledSide | None = None


# A wrap with the operations of one scope it selects.
_Pick = tuple[_Wrap, list[Operation]]


class GraphEdits:
    """The graph edits of one :class:`~neofoam.framework.context.Context`, as ``ctx.graph``.

    Use it from a boundary condition or model that changes how the solver steps,
    e.g. repeats the solution of its field inside a loop, without the solver
    knowing about it. Register edits while the case initializes; the runtime
    applies them once to the resolved graph (:meth:`apply`). Wraps that share one
    ``with_`` operation become one loop around all their operations. Not thread safe.

    Example::

        ctx.graph.wrap(ops=ctx.graph.solving("T"), with_=IterativeOp(three_times))
    """

    def __init__(self) -> None:
        self._wraps: list[_Wrap] = []
        # the coupling of each coupled patch (CouplingOp.shared); a driver that
        # knows the interfaces fills it before the boundary conditions are built
        self.couplings: dict[str, CouplingOp] = {}

    def solving(self, field: str) -> Selector:
        """The operations that solve ``field`` and every operation downstream of them."""
        return Selector(field)

    def wrap(self, ops: Selector, with_: IterativeOp, side: CoupledSide | None = None) -> None:
        """Run the operations ``ops`` selects inside a loop driven by ``with_``.

        The loop takes the place of the last selected operation; the other
        operations of its scope keep their order, those before it run before the
        loop. ``with_`` is called before each pass and repeats the body while it
        returns ``True``. With a ``side``, ``with_`` must be a
        :class:`~neofoam.framework.coupling.CouplingOp`: the side receives before
        the selected operations and sends after them, inside the loop.
        """
        if side is not None:
            if not isinstance(with_, CouplingOp):
                raise TypeError(f"a coupled side needs a CouplingOp, got {type(with_).__name__}")
            with_.attach(ops.field, side)
        self._wraps.append(_Wrap(ops, with_, side))

    def include_region(self, region: GraphEdits, name: str) -> None:
        """Take over the edits of the region ``name``, whose operations this graph steps.

        Use it in a driver that steps a region's operations in its own graph under
        the names ``<name>.<op>`` (:func:`~neofoam.framework.solver.region_operations`).
        The region's wraps then select the region's operations only; wraps of
        several regions that share one ``with_`` join one loop.
        """
        for wrap in region._wraps:
            self._wraps.append(replace(wrap, ops=Selector(f"{name}.{wrap.ops.field}")))

    def apply(self, builder: StepBuilder) -> StepBuilder:
        """Return the resolved graph *builder* with every registered edit applied.

        Raises ``ValueError`` when an edit selects no operation, or selects
        operations in more than one scope.
        """
        if not self._wraps:
            return builder
        applied: set[int] = set()
        ops = _apply_to_scope(builder.operations.ops, self._wraps, applied)
        unused = sorted({wrap.ops.field for wrap in self._wraps if id(wrap) not in applied})
        if unused:
            raise ValueError(
                f"no operation solves {unused}; a graph edit needs an operation "
                "declared with solves=[...] for its field"
            )
        return StepBuilder(ops)


def _apply_to_scope(ops: list[Operation], wraps: list[_Wrap], applied: set[int]) -> list[Operation]:
    """Apply the wraps to the loops nested in *ops*, then to *ops* themselves."""
    ops = [_apply_to_loop(op, wraps, applied) for op in ops]
    for group in _group_by_loop(wraps):
        picks = _picks(ops, group, applied)
        if picks:
            ops = _wrapped(ops, picks, group[0].with_)
    return ops


def _apply_to_loop(op: Operation, wraps: list[_Wrap], applied: set[int]) -> Operation:
    if not isinstance(op.func, IterativeOp):
        return op
    return replace(op, sub_operations=_apply_to_scope(op.sub_operations, wraps, applied))


def _group_by_loop(wraps: list[_Wrap]) -> list[list[_Wrap]]:
    """The wraps grouped by their ``with_`` operation, in registration order."""
    groups: dict[int, list[_Wrap]] = {}
    for wrap in wraps:
        groups.setdefault(id(wrap.with_), []).append(wrap)
    return list(groups.values())


def _picks(ops: list[Operation], group: list[_Wrap], applied: set[int]) -> list[_Pick]:
    """The wraps of *group* that select operations of *ops*, with what they select."""
    picks: list[_Pick] = []
    for wrap in group:
        picked = wrap.ops.select(ops)
        if not picked:
            continue
        if id(wrap) in applied:
            raise ValueError(
                f"the operations solving '{wrap.ops.field}' lie in more than one "
                "loop scope; a graph edit wraps operations of one scope"
            )
        applied.add(id(wrap))
        picks.append((wrap, picked))
    return picks


def _wrapped(ops: list[Operation], picks: list[_Pick], with_: IterativeOp) -> list[Operation]:
    """*ops* with the picked operations moved into one loop at the place of the last one."""
    member_ids = {id(op) for _wrap, picked in picks for op in picked}
    members = [op for op in ops if id(op) in member_ids]
    last = max(index for index, op in enumerate(ops) if id(op) in member_ids)
    solves = sorted({field for op in members for field in op.metadata.solves})
    loop = Operation(
        func=with_,
        metadata=OperationMetadata(op_name=f"loop_solving_{'_'.join(solves)}", solves=solves),
        sub_operations=_with_sides(members, picks),
    )
    # Operations before the last member that are not members do not depend on one
    # (they would be downstream), so running them before the loop keeps every edge.
    earlier = [op for op in ops[:last] if id(op) not in member_ids]
    return [*earlier, loop, *ops[last + 1 :]]


def _with_sides(members: list[Operation], picks: list[_Pick]) -> list[Operation]:
    """*members* with each coupled side's receive before its operations and send after them."""
    before: dict[int, list[Operation]] = {}
    after: dict[int, list[Operation]] = {}
    for wrap, picked in picks:
        if wrap.side is None or not isinstance(wrap.with_, CouplingOp):
            continue
        receive, send = wrap.with_.side_operations(wrap.side, wrap.ops.field)
        before.setdefault(id(picked[0]), []).append(receive)
        after.setdefault(id(picked[-1]), []).append(send)
    body: list[Operation] = []
    for op in members:
        body += [*before.get(id(op), []), op, *after.get(id(op), [])]
    return body
