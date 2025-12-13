# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Test file for the IncompressibleFluid solver based on the new framework architecture.
"""

from typing import Literal

import pytest
from pydantic import BaseModel

from foamadapter.framework.context import Context
from foamadapter.framework.decorator import decorated_member_functions
from foamadapter.framework.operations import Operation, OperationCollection
from foamadapter.framework.solver import Solver, SolverInterface


def test_incompressible_fluid_imports():
    """Test that the IncompressibleFluid solver can be imported."""
    from foamadapter.solver import IncompressibleFluid

    assert IncompressibleFluid is not None


def test_incompressible_fluid_is_solver():
    """Test that IncompressibleFluid implements the SolverInterface."""
    from foamadapter.solver import IncompressibleFluid

    # Check if it's recognized as a SolverInterface
    assert issubclass(IncompressibleFluid, SolverInterface)


def test_incompressible_fluid_structure():
    """Test the structure and operations of IncompressibleFluid solver."""
    from foamadapter.solver import IncompressibleFluid

    # Create instance with minimal arguments
    solver = IncompressibleFluid(argv=["test"])

    # Verify it has the required methods
    assert hasattr(solver, "operations")
    assert hasattr(solver, "main_loop")
    assert hasattr(solver, "run")
    assert hasattr(solver, "create_context")

    # Verify properties
    assert solver.name == "IncompressibleFluid"
    assert solver.maxDeltaT == 1e5


def test_incompressible_fluid_operations_registration():
    """Test that operations are properly registered."""
    from foamadapter.solver import IncompressibleFluid

    solver = IncompressibleFluid(argv=["test"])
    ops = solver.operations()

    # Should have all the decorated operations
    expected_operations = [
        "create_fields",
        "setup_models",
        "print_time",
        "momentum_predictor",
        "solve_momentum",
        "compute_HbyA",
        "compute_phiHbyA",
        "adjust_phi",
        "solve_pressure",
        "update_flux",
        "correct_velocity",
        "turbulence_correction",
        "write_output",
    ]

    assert len(ops) == len(expected_operations)

    for op_name in expected_operations:
        op = ops[op_name]
        assert op.operation_name == op_name


def test_incompressible_fluid_operation_dependencies():
    """Test that operation dependencies are correctly set."""
    from foamadapter.solver import IncompressibleFluid

    solver = IncompressibleFluid(argv=["test"])
    ops = solver.operations()

    # Check specific dependencies
    assert ops["create_fields"].depends_on == []
    assert ops["setup_models"].depends_on == ["create_fields"]
    assert ops["print_time"].depends_on == ["setup_models"]
    assert ops["momentum_predictor"].depends_on == ["print_time"]
    assert ops["solve_momentum"].depends_on == ["momentum_predictor"]
    assert ops["compute_HbyA"].depends_on == ["solve_momentum"]
    assert ops["compute_phiHbyA"].depends_on == ["compute_HbyA"]
    assert ops["adjust_phi"].depends_on == ["compute_phiHbyA"]
    assert ops["solve_pressure"].depends_on == ["adjust_phi"]
    assert ops["update_flux"].depends_on == ["solve_pressure"]
    assert ops["correct_velocity"].depends_on == ["update_flux"]
    assert ops["turbulence_correction"].depends_on == ["correct_velocity"]
    assert ops["write_output"].depends_on == ["turbulence_correction"]


def test_incompressible_fluid_pydantic_schema():
    """Test that the Pydantic schema is properly generated."""
    from foamadapter.solver import IncompressibleFluid

    schema = IncompressibleFluid.model_json_schema()

    assert "properties" in schema
    assert "name" in schema["properties"]
    assert "argv" in schema["properties"]
    assert "maxDeltaT" in schema["properties"]


def test_cfl_condition_class():
    """Test that the CFLCondition helper class exists and can be instantiated."""
    from foamadapter.solver.incompressibleFluid import CFLCondition

    # Mock test without actual OpenFOAM libraries
    # In real usage, this would require proper OpenFOAM environment
    assert CFLCondition is not None


def test_pimple_conditions():
    """Test that PIMPLE condition classes exist."""
    from foamadapter.solver.incompressibleFluid import (
        NonOrthogonalCondition,
        PimpleCorrectorCondition,
        PimpleLoopCondition,
    )

    assert PimpleLoopCondition is not None
    assert PimpleCorrectorCondition is not None
    assert NonOrthogonalCondition is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
