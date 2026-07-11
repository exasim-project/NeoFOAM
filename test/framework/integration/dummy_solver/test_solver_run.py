# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""End-to-end scenario: a full DummySolver run wires the staged init, the
optional-model family, conditional dispatch, and DAG resolution together.

Uses the pybFoam-free DummySolver so the whole LOAD -> RESOLVE -> BUILD ->
execute -> run path can be exercised in-process without OpenFOAM. The
deterministic op formulas let the run assert exact final field values.
"""

from pathlib import Path

import pytest

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
    assert "model_field4" in ctx.fields
    assert ctx.fields["model_field1"] == 300.0
    assert ctx.fields["model_field2"] == 1e-5
    assert ctx.fields["model3_field"] == 0.0
    assert ctx.fields["model_field4"] == 0.0

    # optional_models list contains ModelRuntime objects
    models = ctx.models.get("optional_models", [])
    assert len(models) > 0
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
    assert "model4_step1" in op_names
    assert "solver_step3" in op_names

    idx = {name: op_names.index(name) for name in op_names}
    assert (
        idx["solver_step1"]
        < idx["solver_step2"]
        < idx["model1_step1"]
        < idx["model1_step2"]
        < idx["model2_step1"]
        < idx["model3_step"]
        < idx["model4_step1"]
        < idx["solver_step3"]
    ), f"Operations not sorted by operation_number: {op_names}"


def test_dummy_solver_complete_run() -> None:
    """One full run drives all ops to their exact deterministic final state.

    The inner algorithm loop runs 3 iterations (AlgorithmLoop._max_iterations),
    so every field lands on a fixed value derived from the op formulas — no
    "something moved" checks. Initial values are covered by
    ``test_dummy_solver_full_initialization``.
    """
    from .dummy_solver import run

    ctx_final = run()

    assert ctx_final.fields["field1"] == pytest.approx(0.982096104825)
    assert ctx_final.fields["field2"] == 101305.0
    assert ctx_final.fields["model_field1"] == pytest.approx(300.01993029517496)
    assert ctx_final.fields["model_field3"] == 504.0
    assert ctx_final.fields["model_field4"] == pytest.approx(0.25)
    assert ctx_final.fields["model3_field"] == pytest.approx(4.500199752800874)
    assert ctx_final.models["algorithm"]._iteration_count == 3


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
    rt = spec.instantiate(case_dir=case_dir, instance_id="DummyModel1")

    ops = rt.operations
    assert len(ops) >= 2

    op_names = [op.operation_name for op in ops]
    assert "model1_step1" in op_names
    assert "model1_step2" in op_names

    m1_s1 = next(op for op in ops if op.operation_name == "model1_step1")
    assert str(m1_s1.operation_number) == "2.5"


def test_init_dependency_injection() -> None:
    """@init stages use Depends() for dependency injection."""
    from .dummy_init import create_init

    init = create_init()
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
    rt = spec.instantiate(case_dir=case_dir, instance_id="DummyModel1")

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
