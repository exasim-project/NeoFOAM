# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""Scenario: one spec instantiated as several independent runtimes.

MultiModel (model4) declares no ``@init_runtime`` — it is purely functional.
This file drives multiple runtimes of the same spec to show they share config
values yet carry no per-instance state and never influence one another.
"""

from pathlib import Path

import pytest

from neofoam.framework.context import Context
from .models.model4 import model4, Model4Config


CASE_DIR = Path(__file__).parent / "configs"


def test_model4_instantiate_loads_config() -> None:
    """model4.instantiate() should load Model4Config from disk."""
    rt = model4.instantiate(case_dir=CASE_DIR, instance_id="MultiModel")
    assert rt.config is not None
    assert isinstance(rt.config, Model4Config)
    assert rt.config.scale == 1.5
    assert rt.config.offset == 0.1


def test_model4_two_runtimes_same_config() -> None:
    """Two runtimes from the same spec share config values."""
    rt_a = model4.instantiate(case_dir=CASE_DIR, instance_id="MultiModel_a")
    rt_b = model4.instantiate(case_dir=CASE_DIR, instance_id="MultiModel_b")

    assert rt_a.config.scale == rt_b.config.scale
    assert rt_a.config.offset == rt_b.config.offset


def test_model4_no_instance_state() -> None:
    """No @init_runtime: repeated runs from one runtime give identical results.

    A stateful op would drift between calls; matching values prove no
    per-instance state accumulates on the runtime.
    """
    rt = model4.instantiate(case_dir=CASE_DIR, instance_id="MultiModel")
    op = next(op for op in rt.operations if op.operation_name == "model4_step1")

    def run_once() -> float:
        ctx = Context(fields={"model_field4": 2.0}, models={}, mesh={})
        op.run(ctx)
        return ctx.fields["model_field4"]

    assert run_once() == run_once() == pytest.approx(3.1)


@pytest.mark.parametrize(
    ("model_field4", "expected"),
    [(2.0, 3.1), (1.0, 1.6), (5.0, 7.6)],
)
def test_model4_step_scales_and_offsets(model_field4: float, expected: float) -> None:
    """model4_step1 returns model_field4 * scale(1.5) + offset(0.1)."""
    rt = model4.instantiate(case_dir=CASE_DIR, instance_id="MultiModel")

    ctx = Context(fields={"model_field4": model_field4}, models={}, mesh={})
    op = next(op for op in rt.operations if op.operation_name == "model4_step1")
    op.run(ctx)

    assert ctx.fields["model_field4"] == pytest.approx(expected)


def test_model4_operations_independent() -> None:
    """Operations on rt_a do not affect rt_b (no shared state)."""
    rt_a = model4.instantiate(case_dir=CASE_DIR, instance_id="MultiModel_a")
    rt_b = model4.instantiate(case_dir=CASE_DIR, instance_id="MultiModel_b")

    ctx_a = Context(fields={"model_field4": 1.0}, models={}, mesh={})
    ctx_b = Context(fields={"model_field4": 5.0}, models={}, mesh={})

    op_a = next(op for op in rt_a.operations if op.operation_name == "model4_step1")
    op_b = next(op for op in rt_b.operations if op.operation_name == "model4_step1")

    op_a.run(ctx_a)
    op_b.run(ctx_b)

    assert ctx_a.fields["model_field4"] == pytest.approx(1.6)
    assert ctx_b.fields["model_field4"] == pytest.approx(7.6)
