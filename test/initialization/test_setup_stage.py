# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

"""Tests for SETUP stage."""

from foamadapter.framework.initialization import SolverInitializer
from .test_fixtures import TestSolver, TestTurbulenceModel


def test_setup_stage_marks_methods():
    """Test that @setup decorator marks methods correctly."""
    model = TestTurbulenceModel()

    assert hasattr(model.initialize_fields, "_init_stage")
    assert model.initialize_fields._init_stage == "SETUP"


def test_setup_stage_completes_initialization():
    """Test that SETUP completes all model initialization."""
    solver = TestSolver()
    initializer = SolverInitializer(solver)

    initializer._run_read_files()
    initializer._run_configure()
    initializer._run_setup(mesh=None)

    assert solver.turbulence.setup_complete
    assert solver.transport.setup_complete
    assert solver.algorithm.setup_complete
    assert solver.setup_complete
