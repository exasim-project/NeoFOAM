# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

# NB: no ``from __future__ import annotations`` — Context injection matches
# ``param.annotation is Context`` literally (see field_writer.py).

"""The linear solver residuals of the step, read off the mesh's solverPerformanceDict."""

from typing import Any, ClassVar, Literal

from neofoam.framework.context import Context
from neofoam.postprocess.node import AggregatedData, AggregatedDataSet, Pipeline, Source

#: What labels one residual: the long-format columns before the number.
GROUP_NAMES = ["field", "solver", "metric", "iteration"]

#: The column the number itself fills.
VALUE_COLUMN = "value"

#: The long-format columns in file order; ``time`` is prefixed by the writer.
HEADERS = [*GROUP_NAMES, VALUE_COLUMN]

#: What a field name gains per residual component — nothing for a scalar solve,
#: OpenFOAM's ``x``/``y``/``z`` for a vector one (``Ux``, ``Uy``, ``Uz``).
_COMPONENT_SUFFIXES = {1: ("",), 3: ("x", "y", "z")}

#: Tried in order: the dictionary answers only for the type the field was solved as.
_LOOKUPS = ("lookupSolverPerformanceScalarList", "lookupSolverPerformanceVectorList")


def _components(residual: Any) -> list[float]:
    """The components of one residual — one for a scalar solve, three for a vector one."""
    if hasattr(residual, "__len__"):
        return [float(residual[index]) for index in range(len(residual))]
    return [float(residual)]


def _performances(solver_dict: Any, field: str) -> list[Any]:
    """Every solve of ``field`` recorded this step, whatever type it was solved as.

    The dictionary offers no way to ask which type an entry holds, so the two
    typed lookups are tried in turn and the mismatch they raise is the answer.
    """
    for lookup in _LOOKUPS:
        try:
            return list(getattr(solver_dict, lookup)(field))
        except RuntimeError:  # noqa: PERF203  # the raise *is* the type query
            continue
    raise TypeError(
        f"postProcess residuals: the solves of {field!r} read as neither a scalar nor a "
        "vector performance list; a tensor solve has no residual layout here"
    )


def _rows(field: str, performances: list[Any]) -> list[AggregatedData]:
    """One row per component and metric of every solve of ``field``."""
    rows: list[AggregatedData] = []
    for iteration, performance in enumerate(performances):
        solver = str(performance.solverName())
        for metric, residual in (
            ("initial", performance.initialResidual()),
            ("final", performance.finalResidual()),
        ):
            values = _components(residual)
            for suffix, value in zip(_COMPONENT_SUFFIXES[len(values)], values):
                rows.append(
                    AggregatedData(
                        value=value,
                        group=[f"{field}{suffix}", solver, metric, iteration],
                        group_name=list(GROUP_NAMES),
                    )
                )
    return rows


@Source.register
class Residuals(Source):
    """Every linear solve of the step, in long format — the source that aggregates itself.

    Use it to watch convergence next to the physics tables; there is no field on
    a geometry behind it, so it emits the terminal
    :class:`~neofoam.postprocess.node.AggregatedDataSet` directly and its
    pipeline carries no nodes. Build it through :func:`residuals`::

        residuals()

    ``metric`` is ``initial`` or ``final``, ``iteration`` counts the solves of
    that field within the step (the PIMPLE correctors), and a vector solve
    reports one row per component (``Ux``, ``Uy``, ``Uz``) the way OpenFOAM
    records it. A step that solved nothing yields no rows.

    Decomposed, the rows need no reduction: a linear solve is itself collective,
    so ``solverPerformanceDict`` already holds the same global residuals on every
    rank and the master rank writing alone loses nothing.
    """

    type: Literal["residuals"] = "residuals"

    self_aggregating: ClassVar[bool] = True

    def resolve(self, ctx: Context) -> AggregatedDataSet:
        solver_dict = ctx.mesh.solverPerformanceDict()
        rows: list[AggregatedData] = []
        for name in solver_dict.toc().list():
            field = str(name)
            rows.extend(_rows(field, _performances(solver_dict, field)))
        # named after the column the number fills: every other aggregation names
        # its value column after itself, and this table's is ``value``.
        return AggregatedDataSet(name=VALUE_COLUMN, values=rows)


def residuals() -> Pipeline:
    """Start — and finish — a pipeline from this step's linear-solver residuals.

    The counterpart of :func:`~neofoam.postprocess.sources.fields.field` for solver
    diagnostics; it is already an aggregation, so nothing may be piped onto it::

        residuals()
    """
    return Pipeline(source=Residuals())
