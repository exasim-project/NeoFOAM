# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

"""Tests for LOAD stage."""

import pytest

pytestmark = pytest.mark.skip(
    reason="Outdated - framework refactored (SolverInitializer removed)"
)

# from foamadapter.framework.initialization import SolverInitializer, InitializationStage
# from .initialization_test_models import MockSolver, MockTurbulenceModel


def test_load_stage_marks_methods():
    """Test that @load decorator marks methods correctly."""
    model = MockTurbulenceModel()

    assert hasattr(model.load_coefficients, "_init_stage")
    assert model.load_coefficients._init_stage == InitializationStage.LOAD


def test_load_stage_executes_all_models():
    """Test that LOAD executes on all models."""
    solver = MockSolver()
    initializer = SolverInitializer(solver)

    initializer._run_load()

    assert solver.turbulence.files_read
    assert solver.transport.files_read
    assert solver.algorithm.files_read
    assert solver.files_read


def test_load_loads_correct_data():
    """Test that LOAD loads expected data."""
    solver = MockSolver()
    initializer = SolverInitializer(solver)

    initializer._run_load()

    # Check turbulence loaded coefficients
    assert "C_mu" in solver.turbulence.config.coefficients
    assert solver.turbulence.config.coefficients["C_mu"] == 0.09

    # Check transport loaded properties
    assert solver.transport.config.viscosity == 1e-6
    assert solver.transport.config.density == 998.0
