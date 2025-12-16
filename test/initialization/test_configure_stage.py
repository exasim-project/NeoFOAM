# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

"""Tests for RESOLVE_DEPENDENCIES stage."""

from foamadapter.framework.initialization import SolverInitializer, InitializationStage
from .initialization_test_models import MockSolver, MockTurbulenceModel


def test_resolve_dependencies_stage_marks_methods():
    """Test that @resolve_dependencies decorator marks methods correctly."""
    model = MockTurbulenceModel()

    assert hasattr(model.connect_transport, "_init_stage")
    assert (
        model.connect_transport._init_stage == InitializationStage.RESOLVE_DEPENDENCIES
    )


def test_resolve_dependencies_stage_registers_models():
    """Test that models are registered in registry during RESOLVE_DEPENDENCIES."""
    solver = MockSolver()
    initializer = SolverInitializer(solver)

    initializer._run_load()

    # After load, models should be registered
    assert initializer.config.get("turbulence") is solver.turbulence
    assert initializer.config.get("transport") is solver.transport
    assert initializer.config.get("algorithm") is solver.algorithm


def test_resolve_dependencies_stage_connects_models():
    """Test that models can reference each other during RESOLVE_DEPENDENCIES."""
    solver = MockSolver()
    initializer = SolverInitializer(solver)

    initializer._run_load()
    initializer._run_resolve_dependencies()

    # Turbulence should have reference to transport
    assert solver.turbulence.transport_ref is solver.transport

    # Algorithm should have references to both
    assert solver.algorithm.turbulence_ref is solver.turbulence
    assert solver.algorithm.transport_ref is solver.transport


def test_resolve_dependencies_stage_marks_models_configured():
    """Test that RESOLVE_DEPENDENCIES marks all models as configured."""
    solver = MockSolver()
    initializer = SolverInitializer(solver)

    initializer._run_load()
    initializer._run_resolve_dependencies()

    assert solver.turbulence.configured
    assert solver.transport.configured
    assert solver.algorithm.configured
    assert solver.configured
