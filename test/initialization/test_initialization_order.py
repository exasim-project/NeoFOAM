# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

"""Tests for initialization order and flow."""

from dataclasses import dataclass, field

from foamadapter.framework.model import Model
from foamadapter.framework.solver import Solver
from foamadapter.framework.initialization import SolverInitializer


def test_initialization_order():
    """Test that stages execute in correct order."""
    order = []

    @dataclass
    class TrackingModel:
        name: str = "tracking"
        files_read: bool = False
        configured: bool = False
        setup_complete: bool = False

        @Model.read_files
        def read(self):
            order.append("READ_FILES")
            self.files_read = True

        @Model.configure
        def config(self, registry):
            order.append("CONFIGURE")
            self.configured = True

        @Model.setup
        def set(self, mesh, builder):
            order.append("SETUP")
            self.setup_complete = True

    @dataclass
    class TrackingSolver:
        files_read: bool = False
        configured: bool = False
        setup_complete: bool = False
        model: TrackingModel = field(default_factory=TrackingModel)

        def get_models(self):
            return [self.model]

        @Solver.read_files
        def read(self):
            order.append("SOLVER_READ")
            self.files_read = True

        @Solver.configure
        def config(self, registry):
            order.append("SOLVER_CONFIGURE")
            self.configured = True

        @Solver.setup
        def set(self, mesh, builder):
            order.append("SOLVER_SETUP")
            self.setup_complete = True

    solver = TrackingSolver()
    initializer = SolverInitializer(solver)
    initializer.initialize(mesh=None)

    # Models before solver, stages in order
    expected = [
        "READ_FILES",
        "SOLVER_READ",
        "CONFIGURE",
        "SOLVER_CONFIGURE",
        "SETUP",
        "SOLVER_SETUP",
    ]
    assert order == expected
