# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Tests for DummySolver — complete solver with ModelSpec / ModelRuntime API.
"""

from pathlib import Path

from neofoam.framework.graph import DAGResolver
from neofoam.framework.model import ModelRuntime


def test_dummy_solver_full_initialization() -> None:
    """DummySolver initialization with both core and optional models."""
    from .dummy_solver import dummy_solver as solver

    ctx = solver.initialize()

    # Core fields
    assert "field1" in ctx.fields
    assert "field2" in ctx.fields
    assert "field3" in ctx.fields
    assert ctx.fields["field1"] == 1.0
    assert ctx.fields["field2"] == 101325.0

    # Core models
    assert "algorithm" in ctx.models
    assert "config" in ctx.models
    assert "core2" in ctx.models
    assert ctx.models["core2"].name == "CoreModel2"
    assert ctx.models["core2"].status == "active"
    assert ctx.models["config"]["param1"] == 1e-5
    assert ctx.models["config"]["param2"] == 1000.0

    # Optional model fields
    assert "model_field1" in ctx.fields
    assert "model_field2" in ctx.fields
    assert "model_field3" in ctx.fields
    assert "model3_field" in ctx.fields
    assert "model_field4_instance_a" in ctx.fields
    assert "model_field4_instance_b" in ctx.fields
    assert ctx.fields["model_field1"] == 300.0
    assert ctx.fields["model_field2"] == 1e-5
    assert ctx.fields["model3_field"] == 0.0
    assert ctx.fields["model_field4_instance_a"] == 0.0
    assert ctx.fields["model_field4_instance_b"] == 0.0

    # optional_models list contains ModelRuntime objects
    # 4 specs but MultiModel has 2 instances -> 5 runtimes
    models = ctx.models.get("optional_models", [])
    assert len(models) == 5
    assert all(isinstance(m, ModelRuntime) for m in models)

    # CoupledModel's nested accumulator
    assert "accumulator" in ctx.models


def test_execution_graph_and_dag() -> None:
    """Execution graph structure and DAG resolution."""
    from .dummy_solver import dummy_solver as solver

    solver.initialize()
    builder, model_ops = solver.execution_graph()

    assert len(builder.operations) > 0

    time_loop = builder.operations[0]
    assert time_loop.operation_name == "time_loop"

    inner_loop = time_loop.sub_operations[0]
    assert inner_loop.operation_name == "inner_loop"
    assert len(inner_loop.sub_operations) >= 3

    assert len(model_ops) > 0

    resolver = DAGResolver()
    resolved = resolver.resolve(builder, model_ops)

    inner_ops = resolved.operations[0].sub_operations[0].sub_operations
    op_names = [op.operation_name for op in inner_ops]

    assert "solver_step1" in op_names
    assert "model1_step1" in op_names
    assert "solver_step2" in op_names
    assert "model1_step2" in op_names
    assert "model2_step1" in op_names
    assert "model3_step" in op_names
    assert "model4_step1_instance_a" in op_names
    assert "model4_step1_instance_b" in op_names
    assert "solver_step3" in op_names

    idx = {name: i for i, name in enumerate(op_names)}
    assert (
        idx["solver_step1"]
        < idx["solver_step2"]
        < idx["model1_step1"]
        < idx["model1_step2"]
        < idx["model2_step1"]
        < idx["model3_step"]
    ), f"Operations not sorted by operation_number: {op_names}"

    # Both model4 instance ops should come before solver_step3
    assert idx["model4_step1_instance_a"] < idx["solver_step3"], (
        f"model4_step1_instance_a should be before solver_step3: {op_names}"
    )
    assert idx["model4_step1_instance_b"] < idx["solver_step3"], (
        f"model4_step1_instance_b should be before solver_step3: {op_names}"
    )


def test_dummy_solver_complete_run() -> None:
    """Complete solver run with all operations."""
    from .dummy_solver import dummy_solver as solver, run

    ctx_initial = solver.initialize()
    f1_initial = ctx_initial.fields["field1"]
    f2_initial = ctx_initial.fields["field2"]
    mf1_initial = ctx_initial.fields["model_field1"]
    mf3_initial = ctx_initial.fields["model_field3"]
    mf4a_initial = ctx_initial.fields["model_field4_instance_a"]
    mf4b_initial = ctx_initial.fields["model_field4_instance_b"]

    ctx_final = run()

    assert ctx_final.fields["field1"] != f1_initial
    assert ctx_final.fields["field2"] != f2_initial
    assert ctx_final.fields["model_field1"] != mf1_initial
    assert ctx_final.fields["model_field3"] != mf3_initial
    assert ctx_final.fields["model_field4_instance_a"] != mf4a_initial
    assert ctx_final.fields["model_field4_instance_b"] != mf4b_initial
    assert ctx_final.models["algorithm"]._iteration_count >= 1


def test_dummy_solver_model_operations_executed() -> None:
    """Model operations are discovered and executable via ModelRuntime."""
    from .dummy_solver import dummy_solver as solver

    ctx = solver.initialize()
    models = ctx.models.get("optional_models", [])

    # Find runtimes by spec name
    rt_m1 = next((m for m in models if m.spec.name == "DummyModel1"), None)
    rt_m2 = next((m for m in models if m.spec.name == "DummyModel2"), None)

    assert rt_m1 is not None
    assert rt_m2 is not None

    # Operations are accessible via the runtime
    assert len(rt_m1.operations) > 0
    assert len(rt_m2.operations) > 0

    # Find and call an operation
    m1_step1 = next(
        op for op in rt_m1.operations if op.operation_name == "model1_step1"
    )
    m2_step1 = next(
        op for op in rt_m2.operations if op.operation_name == "model2_step1"
    )

    rt_m1._step1_count = 0
    rt_m2._step1_count = 0

    m1_step1.run(ctx)
    assert rt_m1._step1_count == 1

    m2_step1.run(ctx)
    assert rt_m2._step1_count == 1


def test_model1_operations_discovery() -> None:
    """Operations are auto-discovered from model1 via ModelRuntime."""
    from .models.model1 import model1 as spec

    case_dir = Path(__file__).parent / "configs"
    rt = spec.instantiate(case_dir=case_dir)

    ops = rt.operations
    assert len(ops) >= 2

    op_names = [op.operation_name for op in ops]
    assert "model1_step1" in op_names
    assert "model1_step2" in op_names

    m1_s1 = next(op for op in ops if op.operation_name == "model1_step1")
    assert str(m1_s1.operation_number) == "2.5"


def test_init_dependency_injection() -> None:
    """@init stages use Depends() for dependency injection."""
    from .dummy_init import init

    init.argv = []
    ctx = init.run()

    assert "algorithm" in ctx.models
    assert ctx.models["algorithm"].param1 == 1e-5
    assert "field1" in ctx.fields
    assert ctx.fields["field1"] == 1.0
    assert "field3" in ctx.fields
    assert ctx.fields["field3"] == ctx.fields["field1"] * 0.01


def test_automatic_dependency_injection_from_context() -> None:
    """Solver operations automatically get dependencies from Context."""
    from .dummy_solver import dummy_solver as solver

    ctx = solver.initialize()
    ops = solver.operations

    step1_op = next((op for op in ops if op.operation_name == "solver_step1"), None)
    assert step1_op is not None

    f1_before = ctx.fields["field1"]
    step1_op.run(ctx)
    assert ctx.fields["field1"] != f1_before


def test_model_operations_use_dependency_injection() -> None:
    """Model operations resolve dependencies and build works."""
    from .models.model1 import model1 as spec
    from neofoam.framework.context import Context

    case_dir = Path(__file__).parent / "configs"
    rt = spec.instantiate(case_dir=case_dir)

    lazy_inits = rt.run_build()
    assert len(lazy_inits) == 2
    names = [li.name for li in lazy_inits]
    assert "model_field1" in names
    assert "model_field2" in names
    assert (
        "domain"
        in next(li for li in lazy_inits if li.name == "model_field1").depends_on
    )

    ops = rt.operations
    m1_s1 = next(op for op in ops if op.operation_name == "model1_step1")

    ctx = Context(
        fields={"field1": 2.0, "model_field1": 300.0, "model_field2": 1e-5},
        models={},
        mesh={},
    )
    f1_before = ctx.fields["model_field1"]
    m1_s1.run(ctx)
    assert ctx.fields["model_field1"] > f1_before
