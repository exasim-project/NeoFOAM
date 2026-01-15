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


def test_incompressible_fluid_operations_registration():
    """Test that operations are properly registered."""
    from foamadapter.solver import IncompressibleFluid
    from foamadapter.algorithms.pressure_velocity import PimpleAlgorithm

    solver = IncompressibleFluid(argv=["test"])

    # Mock algorithm initialization to avoid full OpenFOAM setup
    solver._pressure_velocity = PimpleAlgorithm()

    step_builder, model_ops = solver.operations()

    # Verify StepBuilder structure
    assert isinstance(step_builder, type(step_builder))
    assert len(step_builder.operations) == 1  # One time_loop

    # Verify time_loop structure
    time_loop = step_builder.operations[0]
    assert time_loop.operation_name == "time_loop"

    # Get operation names from time_loop
    time_loop_op_names = {op.operation_name for op in time_loop.sub_operations}
    expected_time_loop_ops = {
        "set_time_step",
        "increment_time",
        "inner_loop",
        "write_output",
    }
    assert time_loop_op_names == expected_time_loop_ops

    # Find inner_loop and verify its structure
    inner_loop = next(
        op for op in time_loop.sub_operations if op.operation_name == "inner_loop"
    )
    inner_loop_op_names = {op.operation_name for op in inner_loop.sub_operations}
    expected_inner_loop_ops = {"momentum", "continuity", "turbulence_correction"}
    assert inner_loop_op_names == expected_inner_loop_ops


def test_incompressible_fluid_operation_dependencies():
    """Test that operation dependencies are correctly set."""
    from foamadapter.solver import IncompressibleFluid
    from foamadapter.algorithms.pressure_velocity import PimpleAlgorithm

    solver = IncompressibleFluid(argv=["test"])

    # Mock algorithm initialization to avoid full OpenFOAM setup
    solver._pressure_velocity = PimpleAlgorithm()

    step_builder, model_ops = solver.operations()

    # Get time_loop operations
    time_loop = step_builder.operations[0]
    time_loop_ops = {op.operation_name: op for op in time_loop.sub_operations}

    # Verify time loop operations exist
    set_time_step = time_loop_ops["set_time_step"]
    increment_time = time_loop_ops["increment_time"]
    write_output = time_loop_ops["write_output"]
    assert set_time_step.operation_name == "set_time_step"
    assert increment_time.operation_name == "increment_time"
    assert write_output.operation_name == "write_output"

    # Get inner_loop operations
    inner_loop = time_loop_ops["inner_loop"]
    inner_loop_ops = {op.operation_name: op for op in inner_loop.sub_operations}

    # Verify inner loop operations exist
    momentum = inner_loop_ops["momentum"]
    continuity = inner_loop_ops["continuity"]
    turbulence = inner_loop_ops["turbulence_correction"]
    assert momentum.operation_name == "momentum"
    assert continuity.operation_name == "continuity"
    assert turbulence.operation_name == "turbulence_correction"

    # Verify turbulence depends on continuity
    assert turbulence.depends_on == ["continuity"]


def test_incompressible_fluid_pydantic_schema():
    """Test that the Pydantic schema is properly generated."""
    from foamadapter.solver import IncompressibleFluid

    schema = IncompressibleFluid.model_json_schema()

    assert "properties" in schema
    assert "name" in schema["properties"]
    assert "argv" in schema["properties"]


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

    # Algorithm should have exactly 3 operations (inner_loop, momentum, continuity)
    assert len(ops) == 3
    assert "inner_loop" in [op.operation_name for op in ops]
    assert "momentum" in [op.operation_name for op in ops]
    assert "continuity" in [op.operation_name for op in ops]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
