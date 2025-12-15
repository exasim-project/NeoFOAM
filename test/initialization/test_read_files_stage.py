# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

"""Tests for READ_FILES stage."""

from foamadapter.framework.initialization import SolverInitializer, InitializationStage
from .test_fixtures import TestSolver, TestTurbulenceModel


def test_read_files_stage_marks_methods():
    """Test that @read_files decorator marks methods correctly."""
    model = TestTurbulenceModel()

    assert hasattr(model.load_coefficients, "_init_stage")
    assert model.load_coefficients._init_stage == InitializationStage.READ_FILES


def test_read_files_stage_executes_all_models():
    """Test that READ_FILES executes on all models."""
    solver = TestSolver()
    initializer = SolverInitializer(solver)

    initializer._run_read_files()

    assert solver.turbulence.files_read
    assert solver.transport.files_read
    assert solver.algorithm.files_read
    assert solver.files_read


def test_read_files_loads_correct_data():
    """Test that READ_FILES loads expected data."""
    solver = TestSolver()
    initializer = SolverInitializer(solver)

    initializer._run_read_files()

    # Check turbulence loaded coefficients
    assert "C_mu" in solver.turbulence.config.coefficients
    assert solver.turbulence.config.coefficients["C_mu"] == 0.09

    # Check transport loaded properties
    assert solver.transport.config.viscosity == 1e-6
    assert solver.transport.config.density == 998.0
