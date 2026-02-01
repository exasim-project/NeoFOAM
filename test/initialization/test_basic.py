# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

"""Tests for basic initialization functionality."""

import pytest

pytestmark = pytest.mark.skip(
    reason="Outdated - framework refactored (SolverInitializer removed)"
)

# from foamadapter.framework.initialization import SolverInitializer
# from .initialization_test_models import MockSolver


def test_solver_creation():
    """Test that solver can be created with default models."""
    solver = MockSolver()

    assert solver.turbulence is not None
    assert solver.transport is not None
    assert solver.algorithm is not None


def test_get_models_returns_all_models():
    """Test that get_models returns all embedded models."""
    solver = MockSolver()
    models = solver.get_models()

    assert len(models) == 3
    assert solver.turbulence in models
    assert solver.transport in models
    assert solver.algorithm in models


def test_full_initialization():
    """Test complete 3-stage initialization flow."""
    solver = MockSolver()
    initializer = SolverInitializer(solver)

    result = initializer.initialize(mesh=None)

    # All stages complete
    assert solver.files_read
    assert solver.configured
    assert solver.setup_complete

    # All models initialized
    for model in solver.get_models():
        assert model.files_read
        assert model.configured
        assert model.setup_complete

    # Returns Context (new behavior)
    from foamadapter.framework.context import Context

    assert isinstance(result, Context)
