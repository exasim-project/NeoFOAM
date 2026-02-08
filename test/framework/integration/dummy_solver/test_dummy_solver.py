# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Tests for DummySolver - Complete solver with FastAPI-style API.

Tests the new Solver and Model interface with a complete working example
that doesn't require mocks or external dependencies.
"""

from foamadapter.framework.operations import DAGResolver


def test_dummy_solver_full_initialization():
    """Test DummySolver initialization with both core and optional models."""
    from .dummy_solver import dummy_solver as solver

    ctx = solver.initialize()

    # Verify core fields
    assert "field1" in ctx.fields
    assert "field2" in ctx.fields
    assert "field3" in ctx.fields

    # Verify field values (fields are now scalars)
    assert ctx.fields["field1"] == 1.0
    assert ctx.fields["field2"] == 101325.0

    # Verify core models
    assert "algorithm" in ctx.models
    assert "config" in ctx.models
    assert "core2" in ctx.models

    # Verify core2
    core2 = ctx.models["core2"]
    assert core2.name == "CoreModel2"
    assert core2.status == "active"

    # Verify config
    config = ctx.models["config"]
    assert config["param1"] == 1e-5
    assert config["param2"] == 1000.0

    # Verify optional model fields were created
    assert "model_field1" in ctx.fields
    assert "model_field2" in ctx.fields
    assert "model_field3" in ctx.fields

    # Verify optional model field values
    assert ctx.fields["model_field1"] == 300.0
    assert ctx.fields["model_field2"] == 1e-5
    assert ctx.fields["model_field3"] == 500.0


# ============================================================================
# Test: Execution Graph Building and DAG Resolution
# ============================================================================


def test_execution_graph_and_dag():
    """Test execution graph structure and DAG resolution."""
    from .dummy_solver import dummy_solver as solver

    solver.initialize()

    builder, model_ops = solver.execution_graph()

    # Verify builder has operations
    assert len(builder.operations) > 0

    # Should have time_loop with inner_loop
    time_loop = builder.operations[0]
    assert time_loop.operation_name == "time_loop"
    assert len(time_loop.sub_operations) > 0

    # Inner loop should have solver operations
    inner_loop = time_loop.sub_operations[0]
    assert inner_loop.operation_name == "inner_loop"
    assert len(inner_loop.sub_operations) >= 3  # step1, step2, step3

    # Verify model operations
    assert len(model_ops) > 0  # Should have model1 ops

    # Resolve DAG
    resolver = DAGResolver()
    resolved = resolver.resolve(builder, model_ops)

    # Extract inner loop operations
    time_loop = resolved.operations[0]
    inner_loop = time_loop.sub_operations[0]
    inner_ops = inner_loop.sub_operations

    # Get operation names
    op_names = [op.operation_name for op in inner_ops]

    # Verify all operations present
    assert "solver_step1" in op_names
    assert "model1_step1" in op_names  # From model1
    assert "solver_step2" in op_names
    assert "model1_step2" in op_names  # From model1
    assert "model2_step1" in op_names  # From model2
    assert "solver_step3" in op_names

    # Verify correct order
    idx_step1 = op_names.index("solver_step1")
    idx_m1_step1 = op_names.index("model1_step1")
    idx_step2 = op_names.index("solver_step2")
    idx_m1_step2 = op_names.index("model1_step2")
    idx_m2_step1 = op_names.index("model2_step1")
    idx_step3 = op_names.index("solver_step3")

    # NOTE: Current DAGResolver sorts by operation_number
    # solver_step1 [1.0] < solver_step2 [2.0] < model1_step1 [2.5] < model1_step2 [2.7] < model2_step1 [2.8] < solver_step3 [3.0]
    assert (
        idx_step1 < idx_step2 < idx_m1_step1 < idx_m1_step2 < idx_m2_step1 < idx_step3
    ), f"Operations not sorted by operation_number: {op_names}"

    # Verify dependencies are still correct
    assert idx_step1 < idx_m1_step1  # model1_step1 depends_on solver_step1
    assert idx_m1_step1 < idx_m1_step2  # model1_step2 depends_on model1_step1
    assert idx_step2 < idx_m2_step1  # model2_step1 depends_on solver_step2
    assert idx_step2 < idx_step3  # solver_step3 depends_on solver_step2


# ============================================================================
# ============================================================================
# Test: Complete Solver Run
# ============================================================================


def test_dummy_solver_complete_run():
    """Test complete solver run with all operations."""
    from .dummy_solver import dummy_solver as solver, run

    # Get initial context
    ctx_initial = solver.initialize()

    # Get initial values (fields are now scalars)
    f1_initial = ctx_initial.fields["field1"]
    f2_initial = ctx_initial.fields["field2"]
    mf1_initial = ctx_initial.fields["model_field1"]
    mf3_initial = ctx_initial.fields["model_field3"]

    # Run solver
    ctx_final = run()

    # Verify fields changed
    f1_final = ctx_final.fields["field1"]
    f2_final = ctx_final.fields["field2"]
    mf1_final = ctx_final.fields["model_field1"]
    mf3_final = ctx_final.fields["model_field3"]

    assert f1_final != f1_initial
    assert f2_final != f2_initial
    assert mf1_final != mf1_initial
    assert mf3_final != mf3_initial

    # Verify algorithm ran iterations
    algorithm = ctx_final.models["algorithm"]
    assert algorithm._iteration_count >= 1


def test_dummy_solver_model_operations_executed():
    """Test that model operations are executed during run."""
    from .dummy_solver import dummy_solver as solver

    ctx_initial = solver.initialize()

    # Get models
    models = ctx_initial.models.get("optional_models", [])
    m1 = next((m for m in models if m.name == "DummyModel1"), None)
    m2 = next((m for m in models if m.name == "DummyModel2"), None)

    # Check operations are discovered
    assert len(m1.operations) > 0
    assert len(m2.operations) > 0

    # Reset counters
    m1._step1_count = 0
    m1._step2_count = 0
    m2._step1_count = 0

    # Test m1 step1 can be called
    m1.model1_step1(ctx_initial)
    assert m1._step1_count == 1

    # Test m2 step1 can be called
    m2.model2_step1(ctx_initial)
    assert m2._step1_count == 1


# ============================================================================
# Test: Model Operations Discovery
# ============================================================================


def test_model1_operations_discovery():
    """Test that operations are auto-discovered from model1."""
    from .models.model1 import model1 as model

    # Get operations
    ops = model.operations

    # Verify operations found
    assert len(ops) >= 2

    # Verify operation names
    op_names = [op.operation_name for op in ops]
    assert "model1_step1" in op_names
    assert "model1_step2" in op_names

    # Verify operation numbers
    m1_s1_op = next(op for op in ops if op.operation_name == "model1_step1")
    assert str(m1_s1_op.operation_number) == "2.5"


# ============================================================================
# Test: Dependency Injection via Depends()
# ============================================================================


def test_init_dependency_injection():
    """Test that @init.step uses Depends() for dependency injection."""
    from .dummy_init import init

    # Run init
    init.argv = []
    ctx = init.run()

    # Verify dependencies were resolved
    assert "algorithm" in ctx.models
    assert ctx.models["algorithm"].param1 == 1e-5

    # field1 depends on domain (fields are now scalars)
    assert "field1" in ctx.fields
    assert ctx.fields["field1"] == 1.0  # scalar value

    # field3 depends on field1 (stored as scalar)
    assert "field3" in ctx.fields
    expected_f3 = ctx.fields["field1"] * 0.01
    assert ctx.fields["field3"] == expected_f3


# ============================================================================
# Test: Automatic Dependency Injection
# ============================================================================


def test_automatic_dependency_injection_from_context():
    """Test that operations automatically get dependencies from Context."""
    from .dummy_solver import dummy_solver as solver

    ctx = solver.initialize()

    # Get operations
    ops = solver.operations

    # Find step1 operation
    step1_op = next((op for op in ops if op.operation_name == "solver_step1"), None)
    assert step1_op is not None

    # Run the operation directly with just Context (fields are now scalars)
    f1_before = ctx.fields["field1"]

    # Execute the operation
    step1_op.run(ctx)

    # Verify field1 was updated
    f1_after = ctx.fields["field1"]
    assert f1_after != f1_before


def test_model_operations_use_dependency_injection():
    """Test that model operations automatically resolve dependencies and build works."""
    from .models.model1 import model1 as model
    from pathlib import Path

    # Set up config directory
    case_dir = Path(__file__).parent / "configs"

    # Load configs first (auto-generated load function)
    model.run_load(case_dir=case_dir)

    # Test build creates LazyInit objects
    lazy_inits = model.run_build()
    assert len(lazy_inits) == 2
    names = [li.name for li in lazy_inits]
    assert "model_field1" in names
    assert "model_field2" in names
    f1_init = next(li for li in lazy_inits if li.name == "model_field1")
    assert "domain" in f1_init.depends_on

    # Get operations
    ops = model.operations

    # Find step1 operation
    m1_s1_op = next(op for op in ops if op.operation_name == "model1_step1")
    assert m1_s1_op is not None

    # Create minimal context (fields are now scalars)
    from foamadapter.framework.context import Context

    ctx = Context(
        fields={
            "field1": 2.0,
            "model_field1": 300.0,
            "model_field2": 1e-5,
        },
        models={},
        mesh={},
    )

    f1_before = ctx.fields["model_field1"]

    # Run operation
    m1_s1_op.run(ctx)

    # Verify updated
    f1_after = ctx.fields["model_field1"]
    assert f1_after > f1_before
