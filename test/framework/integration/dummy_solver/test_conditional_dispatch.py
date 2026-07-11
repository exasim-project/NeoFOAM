# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""Scenario: a model whose operation is chosen at RESOLVE time.

CoupledModel (model3) dispatches either ``coupled_step`` (reads another
model's ``model_field1``) or ``standalone_step`` (reads the solver's own
``field1``) based on the ``coupled`` flag that RESOLVE derives from the
presence of DummyModel1. This file exercises both branches plus the nested
Accumulator sub-model, driving the ops directly rather than through a solver
run so the branch and the exact damped values are visible.
"""

from pathlib import Path

import pytest

from neofoam.framework.context import Context
from neofoam.framework.model import ModelRuntime


CASE_DIR = Path(__file__).parent / "configs"


def _make_runtime(coupled: bool) -> ModelRuntime:
    """Runtime for model3 with the ``coupled`` flag RESOLVE would set.

    Loads the real Model3Config from the case dir (``mode``/``damping`` from
    disk) and applies the same ``model_copy(update={"coupled": ...})`` that
    ``model3.resolve`` performs, rather than hand-constructing config values.
    """
    from .models.model3 import model3

    rt = model3.instantiate(CASE_DIR, "CoupledModel")
    rt.config = rt.config.model_copy(update={"coupled": coupled})
    return rt


@pytest.mark.parametrize("coupled", [True, False])
def test_model3_dispatches_single_step(coupled: bool) -> None:
    """Either mode yields exactly one op named model3_step at number 2.9."""
    rt = _make_runtime(coupled=coupled)
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
    """Coupled step reads model_field1 and updates model3_field.

    damped = accumulate(model_field1=300, damping=0.5) = 150;
    model3_field = 0 + 150 * 0.01 = 1.5. A second run from the same runtime
    with a fresh context reproduces 1.5 exactly, proving no instance state.
    """
    from .models.model3 import Accumulator

    rt = _make_runtime(coupled=True)

    def run_once() -> float:
        ctx = Context(
            fields={"model3_field": 0.0, "model_field1": 300.0, "field1": 1.0},
            models={"accumulator": Accumulator()},
            mesh={},
        )
        rt.operations[0].run(ctx)
        return ctx.fields["model3_field"]

    assert run_once() == pytest.approx(1.5)
    assert run_once() == pytest.approx(1.5)


def test_model3_standalone_step_uses_field1() -> None:
    """Standalone step reads field1 (not model_field1).

    damped = accumulate(field1=1.0, damping*0.1=0.05) = 0.05;
    model3_field = 0 + 0.05 * 0.01 = 0.0005. Repeating from the same runtime
    reproduces the value, proving no instance state accumulates.
    """
    from .models.model3 import Accumulator

    rt = _make_runtime(coupled=False)

    def run_once() -> float:
        ctx = Context(
            fields={"model3_field": 0.0, "field1": 1.0},
            models={"accumulator": Accumulator()},
            mesh={},
        )
        rt.operations[0].run(ctx)
        return ctx.fields["model3_field"]

    assert run_once() == pytest.approx(0.0005)
    assert run_once() == pytest.approx(0.0005)


def test_model3_accumulator_tracks_state() -> None:
    """accumulator in ctx.models persists state across operation calls.

    With a shared context and field1=10.0 (damping*0.1=0.05): the first call
    gives 0 + 0.5 * 0.01 = 0.005; the second accumulates to value 1.0, so
    0.005 + 1.0 * 0.01 = 0.015.
    """
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

    assert val1 == pytest.approx(0.005)
    assert val2 == pytest.approx(0.015)
    assert ctx.models["accumulator"].count == 2
