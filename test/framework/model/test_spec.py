# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""Unit tests for :class:`ModelSpec` — the immutable model definition.

Covers the decorator-registration API (``@load``/``@resolve``/``@build``/
``@detect``), field declaration, and ``instantiate()`` — including that
independent runtimes built from one spec never share config or build state.
Registration is asserted through its observable effect (a subsequent
``instantiate``/``run_resolve``/``run_build`` call), not the private
callable storage.
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


def test_model_spec_stores_name() -> None:
    spec = ModelSpec("HeatSource")
    assert spec.name == "HeatSource"


def test_model_factory_returns_model_spec() -> None:
    spec = Model("MyModel")
    assert isinstance(spec, ModelSpec)
    assert spec.name == "MyModel"


def test_spec_load_decorator_registers_loader_used_by_instantiate() -> None:
    spec = ModelSpec("M")

    @spec.load
    def load(case_dir: Any, instance_id: Any) -> Any:
        return {"v": 1}

    rt = spec.instantiate(Path("."), "id")
    assert rt.config == {"v": 1}


def test_spec_resolve_decorator_registers_resolver_used_by_run_resolve() -> None:
    spec = ModelSpec("M")

    @spec.load
    def load(case_dir: Any, instance_id: Any) -> Any:
        return {"v": 1}

    @spec.resolve
    def resolve(config: Any, ctx: Any) -> Any:
        return {"v": config["v"] + 1}

    rt = spec.instantiate(Path("."), "id")
    rt.run_resolve(ctx=SimpleNamespace())  # type: ignore[arg-type]
    assert rt.config == {"v": 2}


def test_spec_build_decorator_registers_builder_used_by_run_build() -> None:
    spec = ModelSpec("M")

    @spec.load
    def load(case_dir: Any, instance_id: Any) -> Any:
        return {}

    @spec.build
    def build(config: Any) -> list[Any]:
        return ["step_a"]

    rt = spec.instantiate(Path("."), "id")
    assert rt.run_build() == ["step_a"]


def test_spec_detect_defaults_to_true() -> None:
    spec = ModelSpec("M")
    assert spec.run_detect() is True


def test_spec_detect_decorator_registers_predicate() -> None:
    spec = ModelSpec("M")

    @spec.detect
    def detect() -> bool:
        return False

    assert spec.run_detect() is False


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
