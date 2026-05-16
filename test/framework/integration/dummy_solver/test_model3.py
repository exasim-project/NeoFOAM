# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Tests for CoupledModel (model3) — rewritten for ModelSpec / ModelRuntime API.
"""

from pathlib import Path


from neofoam.framework.context import Context
from neofoam.framework.model import ModelRuntime


CASE_DIR = Path(__file__).parent / "configs"


def _make_runtime(coupled: bool) -> ModelRuntime:
    """Helper: create a standalone ModelRuntime for model3 with given coupled flag."""
    from .models.model3 import model3, Model3Config

    rt = ModelRuntime(
        spec=model3,
        name="test_coupled" if coupled else "test_standalone",
        config=Model3Config(
            mode="coupled" if coupled else "standalone", damping=0.5, coupled=coupled
        ),
    )
    return rt


def test_model3_operations_dispatch_coupled() -> None:
    """When coupled=True, model3 dispatches coupled_step."""
    rt = _make_runtime(coupled=True)
    ops = rt.operations
    assert len(ops) == 1
    assert ops[0].operation_name == "model3_step"
    assert str(ops[0].operation_number) == "2.9"


def test_model3_operations_dispatch_standalone() -> None:
    """When coupled=False, model3 dispatches standalone_step."""
    rt = _make_runtime(coupled=False)
    ops = rt.operations
    assert len(ops) == 1
    assert ops[0].operation_name == "model3_step"
    assert str(ops[0].operation_number) == "2.9"


def test_model3_build_creates_field_and_accumulator() -> None:
    """build() should produce a field and a nested accumulator model."""
    rt = _make_runtime(coupled=False)
    lazy_inits = rt.run_build()

    names = [li.name for li in lazy_inits]
    assert "model3_field" in names
    assert "accumulator" in names

    field_init = next(li for li in lazy_inits if li.name == "model3_field")
    acc_init = next(li for li in lazy_inits if li.name == "accumulator")
    assert field_init.category == "fields"
    assert acc_init.category == "models"

    from .models.model3 import Accumulator

    acc = acc_init.initializer({})
    assert isinstance(acc, Accumulator)
    assert acc.value == 0.0


def test_model3_config_loaded_via_instantiate() -> None:
    """instantiate() loads Model3Config from disk."""
    from .models.model3 import model3, Model3Config

    rt = model3.instantiate(CASE_DIR, "CoupledModel")
    assert isinstance(rt.config, Model3Config)
    assert rt.config.mode == "coupled"
    assert rt.config.damping == 0.5


def test_model3_coupled_step_uses_model_field1() -> None:
    """Coupled step reads model_field1 and updates model3_field."""
    from .models.model3 import Accumulator

    rt = _make_runtime(coupled=True)

    ctx = Context(
        fields={
            "model3_field": 0.0,
            "model_field1": 300.0,
            "field1": 1.0,
        },
        models={"accumulator": Accumulator()},
        mesh={},
    )

    ops = rt.operations
    ops[0].run(ctx)

    assert ctx.fields["model3_field"] != 0.0
    assert not hasattr(rt, "_step_count")


def test_model3_standalone_step_uses_field1() -> None:
    """Standalone step reads field1 (not model_field1)."""
    from .models.model3 import Accumulator

    rt = _make_runtime(coupled=False)

    ctx = Context(
        fields={"model3_field": 0.0, "field1": 1.0},
        models={"accumulator": Accumulator()},
        mesh={},
    )

    ops = rt.operations
    ops[0].run(ctx)

    assert ctx.fields["model3_field"] != 0.0
    assert not hasattr(rt, "_step_count")


def test_model3_accumulator_tracks_state() -> None:
    """accumulator in ctx.models persists state across operation calls."""
    from .models.model3 import Accumulator

    rt = _make_runtime(coupled=False)
    ctx = Context(
        fields={"model3_field": 0.0, "field1": 10.0},
        models={"accumulator": Accumulator()},
        mesh={},
    )

    ops = rt.operations
    ops[0].run(ctx)
    val1 = ctx.fields["model3_field"]

    ops = rt.operations
    ops[0].run(ctx)
    val2 = ctx.fields["model3_field"]

    assert val2 > val1
    assert ctx.models["accumulator"].count == 2


def test_model3_integrates_with_full_solver() -> None:
    """Model3 operations appear in the resolved DAG and ctx is correctly built."""
    from .dummy_solver import dummy_solver as solver
    from neofoam.framework.operations import OperationCollection

    ctx = solver.initialize()

    assert "model3_field" in ctx.fields
    assert ctx.fields["model3_field"] == 0.0
    assert "accumulator" in ctx.models

    model_ops = OperationCollection()
    for rt in solver.optional_models:
        model_ops.add(rt.operations)

    op_names = [op.operation_name for op in model_ops]
    assert "model3_step" in op_names
