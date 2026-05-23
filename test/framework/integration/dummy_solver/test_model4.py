# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Tests for MultiModel (model4) — multiple instances, no instance state.
"""

from pathlib import Path

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
    """MultiModel has no @init_runtime, so runtimes carry no instance state."""
    rt_a = model4.instantiate(case_dir=CASE_DIR, instance_id="MultiModel_a")
    rt_b = model4.instantiate(case_dir=CASE_DIR, instance_id="MultiModel_b")

    assert not hasattr(rt_a, "_step1_count")
    assert not hasattr(rt_b, "_step1_count")


def test_model4_operation_computes_correctly() -> None:
    """model4_step1 returns model_field4 * scale + offset."""
    rt = model4.instantiate(case_dir=CASE_DIR, instance_id="MultiModel")

    ctx = Context(
        fields={"model_field4": 2.0},
        models={},
        mesh={},
    )

    op = next(op for op in rt.operations if op.operation_name == "model4_step1")
    op.run(ctx)

    expected = 2.0 * 1.5 + 0.1
    assert abs(ctx.fields["model_field4"] - expected) < 1e-9


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

    assert abs(ctx_a.fields["model_field4"] - (1.0 * 1.5 + 0.1)) < 1e-9
    assert abs(ctx_b.fields["model_field4"] - (5.0 * 1.5 + 0.1)) < 1e-9
