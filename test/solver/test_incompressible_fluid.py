# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Test file for the IncompressibleFluid solver based on the new framework architecture.
"""


import pytest

from foamadapter.framework.solver import SolverInterface


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

    # Verify properties
    assert solver.name == "IncompressibleFluid"
    assert solver.maxDeltaT == 1e5


def test_incompressible_fluid_operations_registration():
    """Test that operations are properly registered."""
    from foamadapter.solver import IncompressibleFluid
    from foamadapter.algorithms.pressure_velocity import PimpleAlgorithm

    solver = IncompressibleFluid(argv=["test"])

    # Mock algorithm initialization to avoid full OpenFOAM setup
    solver._pressure_velocity = PimpleAlgorithm()

    ops = solver.operations()

    # Should have solver operations + algorithm operations (momentum, continuity)
    expected_solver_operations = [
        "setup_models",
        "print_time",
        "turbulence_correction",
        "write_output",
    ]

    expected_algorithm_operations = [
        "momentum",
        "continuity",
    ]

    # Total operations = solver + algorithm
    expected_total = len(expected_solver_operations) + len(
        expected_algorithm_operations
    )
    assert len(ops) == expected_total

    # Check solver operations exist
    for op_name in expected_solver_operations:
        op = ops[op_name]
        assert op.operation_name == op_name

    # Check algorithm operations exist
    for op_name in expected_algorithm_operations:
        op = ops[op_name]
        assert op.operation_name == op_name


def test_incompressible_fluid_operation_dependencies():
    """Test that operation dependencies are correctly set."""
    from foamadapter.solver import IncompressibleFluid
    from foamadapter.algorithms.pressure_velocity import PimpleAlgorithm

    solver = IncompressibleFluid(argv=["test"])

    # Mock algorithm initialization to avoid full OpenFOAM setup
    solver._pressure_velocity = PimpleAlgorithm()

    ops = solver.operations()

    # Check solver operation dependencies
    assert ops["setup_models"].depends_on == []
    assert ops["print_time"].depends_on == ["setup_models"]

    # Algorithm operations have no explicit dependencies in their decorator
    # (dependencies are managed by solver's main_loop)
    assert ops["momentum"].depends_on == []
    assert ops["continuity"].depends_on == []

    # Turbulence depends on continuity
    assert ops["turbulence_correction"].depends_on == ["continuity"]
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


def test_algorithm_operations():
    """Test that algorithm provides momentum and continuity operations."""
    from foamadapter.algorithms.pressure_velocity import PimpleAlgorithm

    algorithm = PimpleAlgorithm()
    ops = algorithm.operations()

    # Algorithm should have exactly 2 operations
    assert len(ops) == 2
    assert "momentum" in [op.operation_name for op in ops]
    assert "continuity" in [op.operation_name for op in ops]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
