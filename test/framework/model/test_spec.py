# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Unit tests for ModelSpec / ModelRuntime.

Follows strict TDD: each test exercises one piece of behaviour.
"""

from types import SimpleNamespace
from pathlib import Path
from typing import Any

import pytest

from neofoam.fields.bc import FixedValueBC, NoSlipBC
from neofoam.fields.decl import FieldDecl
from neofoam.fields.value_types import Scalar, Vector
from neofoam.framework.initialization import lazy
from neofoam.framework.model import ModelRuntime, ModelSpec, Model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stub_spec(**kwargs: Any) -> SimpleNamespace:
    """Return a minimal spec-like namespace for testing ModelRuntime in isolation."""
    defaults: dict[str, Any] = {
        "_resolve_func": None,
        "_build_func": None,
        "_build_operations_for": lambda rt: [],
        "_operation_collection_func": None,
        "_operations": [],
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


# ===========================================================================
# Cycle 1 — ModelRuntime: construction and basic properties
# ===========================================================================


def test_model_runtime_stores_name_and_config() -> None:
    spec = _stub_spec()
    rt = ModelRuntime(spec=spec, name="HeatSource_zoneA", config={"power": 100})  # type: ignore[arg-type]
    assert rt.name == "HeatSource_zoneA"
    assert rt.config == {"power": 100}
    assert rt.spec is spec  # type: ignore[comparison-overlap]  # stub is SimpleNamespace


def test_run_resolve_with_no_resolve_func_leaves_config_unchanged() -> None:
    spec = _stub_spec(_resolve_func=None)
    rt = ModelRuntime(spec=spec, name="M_id", config={"v": 1})  # type: ignore[arg-type]
    rt.run_resolve(ctx=SimpleNamespace())  # type: ignore[arg-type]
    assert rt.config == {"v": 1}


def test_run_resolve_calls_spec_func_and_stores_returned_config() -> None:
    new_cfg = {"v": 99}
    spec = _stub_spec(_resolve_func=lambda config, ctx: new_cfg)
    rt = ModelRuntime(spec=spec, name="M_id", config={"v": 1})  # type: ignore[arg-type]
    rt.run_resolve(ctx=SimpleNamespace())  # type: ignore[arg-type]
    assert rt.config is new_cfg


def test_run_resolve_ignores_none_return_and_keeps_original() -> None:
    spec = _stub_spec(_resolve_func=lambda config, ctx: None)
    original = {"v": 1}
    rt = ModelRuntime(spec=spec, name="M_id", config=original)  # type: ignore[arg-type]
    rt.run_resolve(ctx=SimpleNamespace())  # type: ignore[arg-type]
    assert rt.config is original


def test_run_build_returns_empty_list_when_no_build_func() -> None:
    spec = _stub_spec(_build_func=None)
    rt = ModelRuntime(spec=spec, name="M_id", config={})  # type: ignore[arg-type]
    assert rt.run_build() == []


def test_run_build_calls_spec_func_with_config_and_returns_result() -> None:
    captured = {}

    def build(config: Any) -> list[Any]:
        captured["config"] = config
        return ["step_a", "step_b"]

    spec = _stub_spec(_build_func=build)
    rt = ModelRuntime(spec=spec, name="M_id", config={"x": 7})  # type: ignore[arg-type]
    result = rt.run_build()
    assert captured["config"] == {"x": 7}
    assert result == ["step_a", "step_b"]  # type: ignore[comparison-overlap]


def test_operations_delegates_to_spec_build_operations_for() -> None:
    sentinel = object()
    spec = _stub_spec(_build_operations_for=lambda rt: [sentinel])
    rt = ModelRuntime(spec=spec, name="M_id", config={})  # type: ignore[arg-type]
    assert rt.operations == [sentinel]


# ===========================================================================
# Cycle 2 — ModelSpec: decorator API
# ===========================================================================


def test_model_spec_stores_name() -> None:
    spec = ModelSpec("HeatSource")
    assert spec.name == "HeatSource"


def test_model_factory_returns_model_spec() -> None:
    spec = Model("MyModel")
    assert isinstance(spec, ModelSpec)
    assert spec.name == "MyModel"


def test_spec_load_decorator_stores_function() -> None:
    spec = ModelSpec("M")

    @spec.load
    def load(case_dir: Any, instance_id: Any) -> Any:
        return {"v": 1}

    assert spec._load_func is load


def test_spec_resolve_decorator_stores_function() -> None:
    spec = ModelSpec("M")

    @spec.resolve
    def resolve(config: Any, ctx: Any) -> Any:
        return config

    assert spec._resolve_func is resolve


def test_spec_build_decorator_stores_function() -> None:
    spec = ModelSpec("M")

    @spec.build
    def build(config: Any) -> list[Any]:
        return []

    assert spec._build_func is build


def test_spec_detect_defaults_to_true() -> None:
    spec = ModelSpec("M")
    assert spec.run_detect() is True


def test_spec_detect_decorator_stores_and_runs_function() -> None:
    spec = ModelSpec("M")

    @spec.detect
    def detect() -> bool:
        return False

    assert spec.run_detect() is False


# ===========================================================================
# Cycle 3 — ModelSpec.instantiate creates a ModelRuntime
# ===========================================================================


def test_instantiate_without_load_raises() -> None:
    """ModelSpec.instantiate() must raise ValueError if no @load is registered."""
    spec = ModelSpec("M")
    with pytest.raises(ValueError, match="@load"):
        spec.instantiate(Path("."), "instance")


def test_instantiate_calls_load_and_returns_runtime() -> None:
    spec = ModelSpec("HeatSource")

    @spec.load
    def load(case_dir: Any, instance_id: Any) -> Any:
        return {"power": 42, "id": instance_id}

    rt = spec.instantiate(case_dir=Path("."), instance_id="zoneA")

    assert isinstance(rt, ModelRuntime)
    assert rt.name == "HeatSource_zoneA"
    assert rt.config == {"power": 42, "id": "zoneA"}
    assert rt.spec is spec


def test_instantiate_different_ids_produce_independent_runtimes() -> None:
    spec = ModelSpec("HeatSource")

    @spec.load
    def load(case_dir: Any, instance_id: Any) -> Any:
        return {"id": instance_id}

    rt_a = spec.instantiate(Path("."), "zoneA")
    rt_b = spec.instantiate(Path("."), "zoneB")

    assert rt_a.name == "HeatSource_zoneA"
    assert rt_b.name == "HeatSource_zoneB"
    assert rt_a.config is not rt_b.config


# ===========================================================================
# Cycle 4 — State isolation: N runtimes from the same spec never share state
# ===========================================================================


def test_multiple_runtimes_have_independent_configs() -> None:
    """Resolving one runtime must not affect another's config."""
    spec = ModelSpec("M")

    @spec.load
    def load(case_dir: Any, instance_id: Any) -> Any:
        return {"v": 0}

    rt_a = spec.instantiate(Path("."), "a")
    rt_b = spec.instantiate(Path("."), "b")

    # Mutate rt_a's config via resolve
    rt_a.run_resolve(
        ctx=SimpleNamespace(  # type: ignore[arg-type]
            _resolve_func=None,
            all=lambda: {},
        )
    )
    rt_a.config["v"] = 99

    assert rt_b.config["v"] == 0, "rt_b config was corrupted by rt_a mutation"


def test_run_build_results_are_independent_per_runtime() -> None:
    """run_build() on two runtimes with different configs produces different results."""
    spec = ModelSpec("M")

    @spec.load
    def load(case_dir: Any, instance_id: Any) -> Any:
        return {"value": float(instance_id)}

    @spec.build
    def build(config: Any) -> list[Any]:
        return [config["value"]]  # simplistic: return the value as the "step"

    rt_42 = spec.instantiate(Path("."), "42")
    rt_99 = spec.instantiate(Path("."), "99")

    assert rt_42.run_build() == [42.0]  # type: ignore[comparison-overlap]
    assert rt_99.run_build() == [99.0]  # type: ignore[comparison-overlap]


# ===========================================================================
# Cycle 5 — Config injection error cases
# ===========================================================================


def test_find_config_by_type_raises_on_missing_type() -> None:
    """_find_config_by_type raises ValueError when no match exists."""
    from neofoam.framework.config_injection import _find_config_by_type
    from neofoam.io import BaseConfig

    class MyConfig(BaseConfig):
        x: int = 1

    class OtherConfig(BaseConfig):
        y: int = 2

    cfg = MyConfig()
    with pytest.raises(ValueError, match="OtherConfig"):
        _find_config_by_type(cfg, OtherConfig)


def test_find_config_by_type_finds_direct_match() -> None:
    """_find_config_by_type returns the config when it directly matches."""
    from neofoam.framework.config_injection import _find_config_by_type
    from neofoam.io import BaseConfig

    class MyConfig(BaseConfig):
        x: int = 5

    cfg = MyConfig()
    assert _find_config_by_type(cfg, MyConfig) is cfg


def test_find_config_by_type_searches_namespace() -> None:
    """_find_config_by_type locates a config nested inside a SimpleNamespace."""
    from neofoam.framework.config_injection import _find_config_by_type
    from neofoam.io import BaseConfig

    class StepConfig(BaseConfig):
        factor: float = 0.01

    class MainConfig(BaseConfig):
        prop: float = 1.0

    ns = SimpleNamespace(step=StepConfig(), main=MainConfig())
    assert isinstance(_find_config_by_type(ns, StepConfig), StepConfig)
    assert isinstance(_find_config_by_type(ns, MainConfig), MainConfig)


def test_find_config_by_type_raises_when_not_in_namespace() -> None:
    """_find_config_by_type raises when config type missing from namespace."""
    from neofoam.framework.config_injection import _find_config_by_type
    from neofoam.io import BaseConfig

    class Missing(BaseConfig):
        pass

    ns = SimpleNamespace()
    with pytest.raises(ValueError, match="Missing"):
        _find_config_by_type(ns, Missing)


# ===========================================================================
# Cycle 6 — ModelRuntime.configs property
# ===========================================================================


def test_runtime_configs_returns_single_base_config() -> None:
    """When config is a BaseConfig, configs property returns a one-element list."""
    from neofoam.io import BaseConfig

    class MyConfig(BaseConfig):
        v: int = 1

    spec = _stub_spec()
    cfg = MyConfig()
    rt = ModelRuntime(spec=spec, name="rt", config=cfg)  # type: ignore[arg-type]

    assert rt.configs == [cfg]


def test_runtime_configs_returns_all_from_namespace() -> None:
    """When config is a SimpleNamespace of BaseConfigs, all are returned."""
    from neofoam.io import BaseConfig

    class ConfigA(BaseConfig):
        a: int = 1

    class ConfigB(BaseConfig):
        b: int = 2

    spec = _stub_spec()
    ca, cb = ConfigA(), ConfigB()
    rt = ModelRuntime(spec=spec, name="rt", config=SimpleNamespace(a=ca, b=cb))  # type: ignore[arg-type]

    assert sorted(rt.configs, key=id) == sorted([ca, cb], key=id)


def test_runtime_configs_empty_for_non_config() -> None:
    """When config is not a BaseConfig or namespace, configs returns []."""
    spec = _stub_spec()
    rt = ModelRuntime(spec=spec, name="rt", config={"plain": "dict"})  # type: ignore[arg-type]
    assert rt.configs == []

def test_field_returns_decl_handle() -> None:
    spec = Model("Test")
    decl = spec.field(
        "U",
        dimensions=[0, 1, -1, 0, 0, 0, 0],
        value_type=Vector,
        allowed_bcs=[NoSlipBC, FixedValueBC],
        write=True,
    )
    assert isinstance(decl, FieldDecl)
    assert decl.name == "U"
    assert decl.value_type is Vector
    assert decl.allowed_bcs == (NoSlipBC, FixedValueBC)
    assert decl.write is True
    assert decl.depends_on == ("mesh",)


def test_field_decls_lists_declarations_in_order() -> None:
    spec = Model("Test")
    U = spec.field(
        "U",
        dimensions=[0, 1, -1, 0, 0, 0, 0],
        value_type=Vector,
        allowed_bcs=[NoSlipBC],
    )
    p = spec.field(
        "p",
        dimensions=[0, 2, -2, 0, 0, 0, 0],
        value_type=Vector,
        allowed_bcs=[FixedValueBC],
    )
    assert spec.field_decls == (U, p)


def test_duplicate_field_name_is_error() -> None:
    spec = Model("Test")
    spec.field(
        "U",
        dimensions=[0, 1, -1, 0, 0, 0, 0],
        value_type=Vector,
        allowed_bcs=[NoSlipBC],
    )
    with pytest.raises(ValueError, match="field 'U' already declared"):
        spec.field(
            "U",
            dimensions=[0, 1, -1, 0, 0, 0, 0],
            value_type=Vector,
            allowed_bcs=[NoSlipBC],
        )


def test_run_build_auto_synthesizes_for_every_decl() -> None:
    """A bare spec with only field declarations gets a full step list."""
    spec = Model("synth")
    spec.field(
        "U",
        dimensions=[0, 1, -1, 0, 0, 0, 0],
        value_type=Vector,
        allowed_bcs=[NoSlipBC],
        write=True,
    )
    spec.field(
        "p",
        dimensions=[0, 2, -2, 0, 0, 0, 0],
        value_type=Scalar,
        allowed_bcs=[NoSlipBC],
    )

    @spec.build
    def _build() -> list[object]:  # noqa: ANN401
        return []

    rt = ModelRuntime(spec=spec, name="synth", config=None)
    steps = rt.run_build()
    names = [s.name for s in steps]
    assert names == ["fields.U", "fields.p"]
    assert steps[0].write is True
    assert steps[1].write is False


def test_run_build_prepends_field_steps_before_user_steps() -> None:
    """Auto-synthesised steps come first; @build steps follow.

    The topological sort is the source of truth for execution order at
    run time — order in the returned list is only cosmetic. The
    contract this test pins is: a downstream consumer that walks the
    list in order sees fields first, then user-written infrastructure.
    """
    spec = Model("synth_mixed")
    spec.field(
        "U",
        dimensions=[0, 1, -1, 0, 0, 0, 0],
        value_type=Vector,
        allowed_bcs=[NoSlipBC],
    )

    @spec.build
    def _build() -> list[object]:
        return [lazy("control_object", lambda _ctx: None)]

    rt = ModelRuntime(spec=spec, name="synth_mixed", config=None)
    steps = rt.run_build()
    assert [s.name for s in steps] == ["fields.U", "control_object"]


def test_run_build_handles_specs_without_field_decls() -> None:
    """A spec with no Model.field declarations behaves like before."""
    spec = Model("no_fields")

    @spec.build
    def _build() -> list[object]:
        return [lazy("just_a_thing", lambda _ctx: None)]

    rt = ModelRuntime(spec=spec, name="no_fields", config=None)
    steps = rt.run_build()
    assert [s.name for s in steps] == ["just_a_thing"]


def test_run_build_handles_specs_without_build_func() -> None:
    """No @build but declared fields: auto-synthesis still fires."""
    spec = Model("only_fields")
    spec.field(
        "T",
        dimensions=[0, 0, 0, 1, 0, 0, 0],
        value_type=Scalar,
        allowed_bcs=[NoSlipBC],
    )
    rt = ModelRuntime(spec=spec, name="only_fields", config=None)
    steps = rt.run_build()
    assert [s.name for s in steps] == ["fields.T"]
