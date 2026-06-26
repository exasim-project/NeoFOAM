# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for the solution-loop gather-point interfaces."""

# NOTE: no `from __future__ import annotations` — keep annotations live.

import ast
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

import pytest

import neofoam.algorithms.solution_loop as loop_pkg
from neofoam.algorithms.solution_loop.interfaces import (
    VGREAT,
    loopCondition,
    timeStepConstraint,
)
from neofoam.framework.context import Context
from neofoam.framework.interface import InterfaceSpec


@contextmanager
def _temporary_contributions(
    spec: InterfaceSpec[object], fns: list[Callable[[], object]]
) -> Iterator[None]:
    """Register *fns* on *spec* for the duration of the block, then remove them."""
    for fn in fns:
        spec.contribute(fn)
    try:
        yield
    finally:
        for fn in fns:
            spec._contributions.remove(fn)
            spec._owner.pop(fn, None)


def _ctx() -> Context:
    return Context(fields={}, models={})


@pytest.mark.parametrize(
    "limits, expected",
    [
        ([], VGREAT),
        ([lambda: 1.0, lambda: 2.0], 1.0),
        ([lambda: 2.0, lambda: 1.0, lambda: 3.0], 1.0),  # order-independent
    ],
)
def test_constraint_fold_min_over_active_contributions(
    limits: list[Callable[[], object]], expected: float
) -> None:
    with _temporary_contributions(timeStepConstraint, limits):
        assert timeStepConstraint.collect(_ctx()) == expected


@pytest.mark.parametrize(
    "flags, expected",
    [
        ([], True),
        ([lambda: True, lambda: True], True),
        ([lambda: True, lambda: False], False),
    ],
)
def test_condition_fold_all_over_active_contributions(
    flags: list[Callable[[], object]], expected: bool
) -> None:
    with _temporary_contributions(loopCondition, flags):
        assert loopCondition.collect(_ctx()) is expected


def _imported_names(source: str) -> set[str]:
    """Return every top-level module name an import statement references."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_framework_solution_loop_imports_no_pybfoam() -> None:
    # The framework loop must stay pure-Python: no module under the loop package
    # may import pybFoam, so the backend-typed CFL contribution can only live
    # solver-side. (A sys.modules check is unusable here because the eager
    # neofoam package root pulls pybFoam via the unrelated io/solver stack.)
    pkg_dir = Path(loop_pkg.__file__).parent
    offenders = {
        path.name: imported
        for path in pkg_dir.glob("*.py")
        if "pybFoam" in (imported := _imported_names(path.read_text()))
    }
    assert not offenders, offenders
