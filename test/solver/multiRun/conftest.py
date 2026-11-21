# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""multiRun resolves the test-only solvers by name; they are no package of neofoam.solver."""

import importlib
from collections.abc import Iterator
from typing import Any

import pytest

from solver.multiRun.probe_solver import probe
from solver.multiRun.slab_solver import slab
from solver.multiRun.steady_solver import steady

MULTI_RUN = importlib.import_module("neofoam.solver.multiRun.multiRun")
SOLVERS = {"probe": probe, "slab": slab, "steady": steady}


@pytest.fixture(scope="package", autouse=True)
def _test_solvers() -> Iterator[None]:
    resolve = MULTI_RUN.resolve_solver

    def resolve_test_solver(name: str) -> Any:
        return SOLVERS[name] if name in SOLVERS else resolve(name)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(MULTI_RUN, "resolve_solver", resolve_test_solver)
        yield
