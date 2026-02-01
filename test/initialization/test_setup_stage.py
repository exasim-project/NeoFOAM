# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

"""Tests for BUILD stage."""

import pytest

pytestmark = pytest.mark.skip(reason="Outdated - framework refactored (SolverInitializer removed)")

# from foamadapter.framework.initialization import SolverInitializer, InitializationStage
# from .initialization_test_models import MockSolver, MockTurbulenceModel


def test_build_stage_marks_methods():
    """Test that @build decorator marks methods correctly."""
    model = MockTurbulenceModel()

    assert hasattr(model.initialize_fields, "_init_stage")
    assert model.initialize_fields._init_stage == InitializationStage.BUILD


def test_build_stage_completes_initialization():
    """Test that BUILD completes all model initialization."""
    solver = MockSolver()
    initializer = SolverInitializer(solver)

    initializer._run_load()
    initializer._run_resolve_dependencies()
    initializer._run_build(mesh=None)

    assert solver.turbulence.setup_complete
    assert solver.transport.setup_complete
    assert solver.algorithm.setup_complete
    assert solver.setup_complete
