# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Tests for MultiModel (model4) — multi-instance discovery from YAML config.
"""

from pathlib import Path

from neofoam.framework.context import Context
from .models.model4 import model4, Model4Config


CASE_DIR = Path(__file__).parent / "configs"


def test_model4_detect_returns_instance_ids() -> None:
    """detect should return instance IDs from config file keys."""
    result = model4.run_detect(case_dir=CASE_DIR)
    assert result.detected
    assert set(result.instance_ids) == {"instance_a", "instance_b"}


def _entry(name: str) -> dict[str, str]:
    return {"type": "MultiModel", "name": name}


def test_model4_instance_a_loads_config() -> None:
    """instance_a should load scale=1.5, offset=0.1."""
    rt = model4.instantiate(case_dir=CASE_DIR, entry=_entry("instance_a"))
    assert isinstance(rt.config, Model4Config)
    assert rt.config.scale == 1.5
    assert rt.config.offset == 0.1


def test_model4_instance_b_loads_config() -> None:
    """instance_b should load scale=2.0, offset=0.5."""
    rt = model4.instantiate(case_dir=CASE_DIR, entry=_entry("instance_b"))
    assert isinstance(rt.config, Model4Config)
    assert rt.config.scale == 2.0
    assert rt.config.offset == 0.5


def test_model4_instances_independent_configs() -> None:
    """Two instances have different config values."""
    rt_a = model4.instantiate(case_dir=CASE_DIR, entry=_entry("instance_a"))
    rt_b = model4.instantiate(case_dir=CASE_DIR, entry=_entry("instance_b"))

    assert rt_a.config.scale != rt_b.config.scale
    assert rt_a.config.offset != rt_b.config.offset


def test_model4_operation_computes_correctly_instance_a() -> None:
    """model4_step1 for instance_a uses model_field4_instance_a."""
    rt = model4.instantiate(case_dir=CASE_DIR, entry=_entry("instance_a"))

    ctx = Context(
        fields={"model_field4_instance_a": 2.0},
        models={},
        mesh={},
    )

    op = next(
        op for op in rt.operations if op.operation_name == "model4_step1_instance_a"
    )
    op.run(ctx)

    expected = 2.0 * 1.5 + 0.1
    assert abs(ctx.fields["model_field4_instance_a"] - expected) < 1e-9


def test_model4_operation_computes_correctly_instance_b() -> None:
    """model4_step1 for instance_b uses model_field4_instance_b."""
    rt = model4.instantiate(case_dir=CASE_DIR, entry=_entry("instance_b"))

    ctx = Context(
        fields={"model_field4_instance_b": 3.0},
        models={},
        mesh={},
    )

    op = next(
        op for op in rt.operations if op.operation_name == "model4_step1_instance_b"
    )
    op.run(ctx)

    expected = 3.0 * 2.0 + 0.5
    assert abs(ctx.fields["model_field4_instance_b"] - expected) < 1e-9


def test_model4_operations_independent() -> None:
    """Operations on instance_a do not affect instance_b."""
    rt_a = model4.instantiate(case_dir=CASE_DIR, entry=_entry("instance_a"))
    rt_b = model4.instantiate(case_dir=CASE_DIR, entry=_entry("instance_b"))

    ctx = Context(
        fields={"model_field4_instance_a": 1.0, "model_field4_instance_b": 5.0},
        models={},
        mesh={},
    )

    op_a = next(
        op for op in rt_a.operations if op.operation_name == "model4_step1_instance_a"
    )
    op_b = next(
        op for op in rt_b.operations if op.operation_name == "model4_step1_instance_b"
    )

    op_a.run(ctx)
    op_b.run(ctx)

    assert abs(ctx.fields["model_field4_instance_a"] - (1.0 * 1.5 + 0.1)) < 1e-9
    assert abs(ctx.fields["model_field4_instance_b"] - (5.0 * 2.0 + 0.5)) < 1e-9


# ===========================================================================
# Manifest-based instantiation
# ===========================================================================


def test_model4_manifest_instantiation() -> None:
    """Model4 can be instantiated from a manifest entry (no @load needed)."""
    from neofoam.framework.model import load_manifest

    manifest_path = CASE_DIR / "models.yaml"
    runtimes = load_manifest(manifest_path, CASE_DIR, "DummyModelInterface")

    assert len(runtimes) == 2
    by_name = {rt.name: rt for rt in runtimes}
    assert "instance_a" in by_name
    assert "instance_b" in by_name

    rt_a = by_name["instance_a"]
    assert isinstance(rt_a.config, Model4Config)
    assert rt_a.config.scale == 1.5
    assert rt_a.config.offset == 0.1

    rt_b = by_name["instance_b"]
    assert rt_b.config.scale == 2.0
    assert rt_b.config.offset == 0.5


def test_model4_config_auto_construct() -> None:
    """When entry has inline config fields, @config auto-constructs without @load."""
    from neofoam.framework.model import ModelSpec
    from neofoam.io import BaseConfig

    spec = ModelSpec("TestAutoConfig")

    @spec.config
    class Cfg(BaseConfig):
        scale: float = 0.0
        offset: float = 0.0

    entry = {"type": "TestAutoConfig", "name": "auto_inst", "scale": 3.0, "offset": 0.5}
    rt = spec.instantiate(case_dir=CASE_DIR, entry=entry)
    assert rt.name == "auto_inst"
    assert rt.config.scale == 3.0
    assert rt.config.offset == 0.5
