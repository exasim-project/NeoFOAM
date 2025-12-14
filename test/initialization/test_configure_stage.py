# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

"""Tests for CONFIGURE stage."""

from foamadapter.framework.initialization import SolverInitializer
from .test_fixtures import TestSolver, TestTurbulenceModel


def test_configure_stage_marks_methods():
    """Test that @configure decorator marks methods correctly."""
    model = TestTurbulenceModel()
    
    assert hasattr(model.connect_transport, "_init_stage")
    assert model.connect_transport._init_stage == "CONFIGURE"


def test_configure_stage_registers_models():
    """Test that models are registered in registry during CONFIGURE."""
    solver = TestSolver()
    initializer = SolverInitializer(solver)
    
    initializer._run_read_files()
    
    # After read_files, models should be registered
    assert initializer.registry.get("turbulence") is solver.turbulence
    assert initializer.registry.get("transport") is solver.transport
    assert initializer.registry.get("algorithm") is solver.algorithm


def test_configure_stage_connects_models():
    """Test that models can reference each other during CONFIGURE."""
    solver = TestSolver()
    initializer = SolverInitializer(solver)
    
    initializer._run_read_files()
    initializer._run_configure()
    
    # Turbulence should have reference to transport
    assert solver.turbulence.transport_ref is solver.transport
    
    # Algorithm should have references to both
    assert solver.algorithm.turbulence_ref is solver.turbulence
    assert solver.algorithm.transport_ref is solver.transport


def test_configure_stage_marks_models_configured():
    """Test that CONFIGURE marks all models as configured."""
    solver = TestSolver()
    initializer = SolverInitializer(solver)
    
    initializer._run_read_files()
    initializer._run_configure()
    
    assert solver.turbulence.configured
    assert solver.transport.configured
    assert solver.algorithm.configured
    assert solver.configured
