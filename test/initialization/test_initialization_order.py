# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

"""Tests for initialization order and flow."""

import pytest

pytestmark = pytest.mark.skip(
    reason="Outdated - framework refactored (SolverInitializer removed)"
)

# from dataclasses import dataclass, field
#
# from foamadapter.framework.model import Model
# from foamadapter.framework.solver import Solver
# from foamadapter.framework.initialization import SolverInitializer


def test_initialization_order():
    """Test that stages execute in correct order."""
    order = []

    @dataclass
    class TrackingModel:
        name: str = "tracking"
        files_read: bool = False
        configured: bool = False
        setup_complete: bool = False

        @Model.load
        def read(self):
            order.append("LOAD")
            self.files_read = True

        @Model.resolve_dependencies
        def config(self, registry):
            order.append("RESOLVE_DEPENDENCIES")
            self.configured = True

        @Model.build
        def set(self, mesh):
            order.append("BUILD")
            self.setup_complete = True
            return []  # Return empty list of lazy initializers

    @dataclass
    class TrackingSolver:
        files_read: bool = False
        configured: bool = False
        setup_complete: bool = False
        model: TrackingModel = field(default_factory=TrackingModel)

        def get_models(self):
            return [self.model]

        @Solver.load
        def read(self):
            order.append("SOLVER_LOAD")
            self.files_read = True

        @Solver.resolve_dependencies
        def config(self, registry):
            order.append("SOLVER_RESOLVE_DEPENDENCIES")
            self.configured = True

        @Solver.build
        def set(self, mesh):
            order.append("SOLVER_BUILD")
            self.setup_complete = True
            return []  # Return empty list of lazy initializers

    solver = TrackingSolver()
    initializer = SolverInitializer(solver)
    initializer.initialize(mesh=None)

    # Models before solver, stages in order
    expected = [
        "LOAD",
        "SOLVER_LOAD",
        "RESOLVE_DEPENDENCIES",
        "SOLVER_RESOLVE_DEPENDENCIES",
        "BUILD",
        "SOLVER_BUILD",
    ]
    assert order == expected
