# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""End-to-end: the ``residuals`` source against a real solverPerformanceDict.

**Integration, real OpenFOAM.** ``test_residuals.py`` pins the row layout against
a faked dictionary; only a solved case can pin what OpenFOAM actually records —
that ``p`` is a scalar solve repeated once per pressure corrector and ``U`` a
single vector solve reported per component. Building and solving the case is
:func:`solver.incompressibleFluid.solved_case.solved_cavity`, the same cavity
``test/solver/incompressibleFluid/test_post_process.py`` runs, in a fresh
interpreter for the same reason (one ``Foam::Time`` per process).

**No numeric expectation.** A residual is solver- and machine-dependent, so the
assertions are the ones that hold for every run: the header, non-negativity, and
which fields OpenFOAM names.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from solver.incompressibleFluid.solved_case import read_table, solved_cavity

from neofoam.tooling.casebuild import CaseDir

_DECLARATION = Path(__file__).parents[1] / "cases" / "residuals"


@pytest.fixture(scope="module")
def cavity_with_residuals(tmp_path_factory: pytest.TempPathFactory) -> CaseDir:
    """The cavity solved for three steps with the ``residuals`` table declared."""
    dest = tmp_path_factory.mktemp("residuals") / "cavity"
    return solved_cavity(dest, declarations_dir=_DECLARATION)


@pytest.fixture(scope="module")
def residual_rows(cavity_with_residuals: CaseDir) -> list[list[str]]:
    """The data rows of ``postProcessing/residuals.csv``, header dropped."""
    _, rows = read_table(cavity_with_residuals, "residuals")
    return rows


def test_the_table_is_written_in_long_format(cavity_with_residuals: CaseDir) -> None:
    header, _ = read_table(cavity_with_residuals, "residuals")

    assert header == ["time", "field", "solver", "metric", "iteration", "value"]


def test_every_write_step_contributes_rows(residual_rows: list[list[str]]) -> None:
    assert sorted({float(row[0]) for row in residual_rows}) == pytest.approx([0.001, 0.002, 0.003])


def test_every_residual_is_non_negative(residual_rows: list[list[str]]) -> None:
    assert all(float(row[5]) >= 0.0 for row in residual_rows)


def test_the_pressure_and_every_velocity_component_are_reported(
    residual_rows: list[list[str]],
) -> None:
    assert {row[1] for row in residual_rows} == {"p", "Ux", "Uy", "Uz"}


def test_both_metrics_are_recorded_for_every_inner_iteration(
    residual_rows: list[list[str]],
) -> None:
    pressure = [row for row in residual_rows if row[1] == "p" and row[0] == "0.003"]

    assert [(row[3], row[4]) for row in pressure] == [
        ("initial", "0"),
        ("final", "0"),
        ("initial", "1"),
        ("final", "1"),
    ]
