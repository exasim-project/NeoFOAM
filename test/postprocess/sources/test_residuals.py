# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The ``residuals`` source: the row layout it derives from a solverPerformanceDict.

The dictionary is the only thing faked, and it is faked exactly as pybFoam
exposes it (verified on a solved cavity): ``toc().list()`` names the fields, a
typed ``lookupSolverPerformance<T>List`` answers for the type the field was
solved as and raises ``RuntimeError`` for every other type, and a vector solve
reports a three-component residual. That keeps the layout — one row per field
component, metric and inner iteration — provable without a solver run;
``test_residuals_e2e.py`` pins it against the real dictionary.

The rows are asserted in *file* order (``table_headers`` / ``table_rows``: what
labels a residual, then the number), because that long format is the contract a
reader of ``residuals.csv`` has.
"""

from __future__ import annotations

from typing import Any

import pytest

from neofoam.framework.context import Context
from neofoam.postprocess.node import AggregatedDataSet
from neofoam.postprocess.sources.residuals import HEADERS, Residuals, residuals
from neofoam.postprocess.writers.writer import table_headers, table_rows


class FakePerformance:
    """One recorded solve: a solver name and an initial/final residual."""

    def __init__(self, solver: str, initial: Any, final: Any) -> None:
        self._solver = solver
        self._initial = initial
        self._final = final

    def solverName(self) -> str:
        return self._solver

    def initialResidual(self) -> Any:
        return self._initial

    def finalResidual(self) -> Any:
        return self._final


class FakeWordList:
    """The dictionary's table of contents, as pybFoam's ``wordList`` exposes it."""

    def __init__(self, names: list[str]) -> None:
        self._names = names

    def list(self) -> list[str]:
        return self._names


class FakeSolverPerformanceDict:
    """The mesh's performance dict: a typed lookup raises for the wrong type."""

    def __init__(self, **by_type: dict[str, list[FakePerformance]]) -> None:
        self._by_type = by_type

    def toc(self) -> FakeWordList:
        return FakeWordList([name for entries in self._by_type.values() for name in entries])

    def _lookup(self, kind: str, field: str) -> list[FakePerformance]:
        entries = self._by_type.get(kind, {})
        if field not in entries:
            raise RuntimeError(f"Entry {field} could not be read as List<{kind}>")
        return entries[field]

    def lookupSolverPerformanceScalarList(self, field: str) -> list[FakePerformance]:
        return self._lookup("Scalar", field)

    def lookupSolverPerformanceVectorList(self, field: str) -> list[FakePerformance]:
        return self._lookup("Vector", field)


class FakeMesh:
    """An fvMesh: the residuals source reads only the performance dict off it."""

    def __init__(self, solver_dict: FakeSolverPerformanceDict) -> None:
        self._solver_dict = solver_dict

    def solverPerformanceDict(self) -> FakeSolverPerformanceDict:
        return self._solver_dict


def _ctx(solver_dict: FakeSolverPerformanceDict) -> Context:
    return Context(models={}, fields={}, mesh=FakeMesh(solver_dict))


def test_a_scalar_field_gets_one_row_per_solve_and_metric() -> None:
    solver_dict = FakeSolverPerformanceDict(
        Scalar={
            "p": [FakePerformance("DICPCG", 1.0, 0.03), FakePerformance("DICPCG", 0.017, 4e-08)]
        }
    )

    result = Residuals().resolve(_ctx(solver_dict))

    assert table_headers(result) == HEADERS
    assert table_rows(result) == [
        ["p", "DICPCG", "initial", 0, 1.0],
        ["p", "DICPCG", "final", 0, 0.03],
        ["p", "DICPCG", "initial", 1, 0.017],
        ["p", "DICPCG", "final", 1, 4e-08],
    ]


def test_a_vector_solve_reports_one_row_per_component() -> None:
    solver_dict = FakeSolverPerformanceDict(
        Vector={"U": [FakePerformance("DILUPBiCGStab", (0.1, 0.2, 0.0), (1e-9, 2e-9, 0.0))]}
    )

    result = Residuals().resolve(_ctx(solver_dict))

    assert table_rows(result) == [
        ["Ux", "DILUPBiCGStab", "initial", 0, 0.1],
        ["Uy", "DILUPBiCGStab", "initial", 0, 0.2],
        ["Uz", "DILUPBiCGStab", "initial", 0, 0.0],
        ["Ux", "DILUPBiCGStab", "final", 0, 1e-09],
        ["Uy", "DILUPBiCGStab", "final", 0, 2e-09],
        ["Uz", "DILUPBiCGStab", "final", 0, 0.0],
    ]


def test_a_step_that_solved_nothing_yields_no_rows() -> None:
    # the columns of a row are only known once there is one, so an empty step
    # names none of them and the writer leaves the file alone until the next
    result = Residuals().resolve(_ctx(FakeSolverPerformanceDict()))

    assert result.values == []
    assert table_rows(result) == []


def test_a_solve_of_an_unreadable_type_names_the_field() -> None:
    solver_dict = FakeSolverPerformanceDict(Tensor={"R": [FakePerformance("PBiCG", 1.0, 0.0)]})

    with pytest.raises(TypeError, match=r"'R'.*neither a scalar nor a vector"):
        Residuals().resolve(_ctx(solver_dict))


def test_the_sugar_is_a_pipeline_with_no_nodes_that_already_aggregates() -> None:
    solver_dict = FakeSolverPerformanceDict(Scalar={"p": [FakePerformance("DICPCG", 1.0, 0.03)]})
    pipeline = residuals()

    result = pipeline.compute(_ctx(solver_dict))

    assert pipeline.steps == []
    assert isinstance(result, AggregatedDataSet)
