# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

"""Unit tests for initialization helpers."""

from typing import Any, Callable

import pytest

from neofoam.framework.initialization.execution.executor import execute_step
from neofoam.framework.initialization.helpers import (
    field,
    operator,
    lazy,
    model,
    InitializerBuilder,
)
from neofoam.framework.initialization.init_step import InitStep


# --- Helper function tests (parametrized) ---


@pytest.mark.parametrize(
    "helper,prefix,category",
    [
        (field, "fields.", "fields"),
        (operator, "operators.", "operators"),
        (model, "models.", "models"),
        (lazy, "", "resource"),
    ],
)
def test_helper_naming(
    helper: Callable[..., InitStep], prefix: str, category: str
) -> None:
    """All helpers create InitStep with correct name prefix and category."""
    result = helper("X", create=lambda _ctx: 1)
    assert isinstance(result, InitStep)
    assert result.name == f"{prefix}X"
    assert result.category == category
    assert result.depends_on == []


@pytest.mark.parametrize("helper", [field, operator, model, lazy])
def test_helper_with_deps(helper: Callable[..., InitStep]) -> None:
    """All helpers pass through custom depends_on."""
    result = helper("X", create=lambda _ctx: 1, depends_on=["a", "b"])
    assert result.depends_on == ["a", "b"]


def test_helper_execute() -> None:
    """Helpers produce executable InitStep objects."""
    assert execute_step(field("U", create=lambda _ctx: "velocity"), {}) == "velocity"
    assert execute_step(lazy("mesh", create=lambda _ctx: "mesh_obj"), {}) == "mesh_obj"


# --- InitializerBuilder tests ---


@pytest.mark.parametrize(
    "method,expected_prefix,kwargs",
    [
        (
            InitializerBuilder.add_field,
            "fields.",
            {"depends_on": ["mesh"], "value": 42},
        ),
        (InitializerBuilder.add_model, "models.", {"value": "instance"}),
        (
            InitializerBuilder.add_operator,
            "operators.",
            {"depends_on": ["fields.U"], "value": "op"},
        ),
    ],
    ids=["add_field", "add_model", "add_operator"],
)
def test_builder_add_methods(
    builder: InitializerBuilder,
    method: Callable[..., Any],
    expected_prefix: str,
    kwargs: dict[str, Any],
) -> None:
    """Builder add_field/add_model/add_operator create correctly prefixed InitStep."""
    method(builder, "X", **kwargs)
    inits = builder.build()
    assert len(inits) == 1
    assert inits[0].name == f"{expected_prefix}X"


def test_builder_add_resource(builder: InitializerBuilder) -> None:
    """add_resource creates an unprefixed InitStep."""
    builder.add_resource("mesh", "mock_mesh")
    inits = builder.build()
    assert len(inits) == 1
    assert inits[0].name == "mesh"
    assert execute_step(inits[0], {}) == "mock_mesh"


def test_builder_add_field_callable(builder: InitializerBuilder) -> None:
    """add_field with callable value passes the callable through."""
    builder.add_field("p", depends_on=["mesh"], value=lambda _ctx: "computed")
    assert execute_step(builder.build()[0], {}) == "computed"


def test_builder_add_model_callable(builder: InitializerBuilder) -> None:
    """add_model with callable value passes the callable through."""
    builder.add_model("turb", value=lambda _ctx: "turb_inst")
    assert execute_step(builder.build()[0], {}) == "turb_inst"


def test_builder_chaining(builder: InitializerBuilder) -> None:
    """All builder methods return self for chaining."""
    result = (
        builder.add_resource("mesh", "m")
        .add_field("U", depends_on=["mesh"], value=1)
        .add_model("algo", value=2)
        .add_operator("mom", depends_on=[], value=3)
    )
    assert result is builder
    assert len(builder.build()) == 4


def test_builder_add_core_models(
    builder: InitializerBuilder, mock_core_model: Any
) -> None:
    """add_core_models adds model + InitSteps from run_build() unchanged."""
    builder.add_core_models([("algorithm", mock_core_model)])
    inits = builder.build()

    assert len(inits) >= 3
    model_inits = [li for li in inits if li.name == "models.algorithm"]
    assert len(model_inits) == 1
    assert any(li.name == "test_field" and li.category == "fields" for li in inits)
    assert any(li.name == "test_op" and li.category == "operators" for li in inits)


def test_builder_add_core_models_without_name(builder: InitializerBuilder) -> None:
    """add_core_models uses lowercase class name when no tuple name given."""

    class AlgorithmModel:
        def run_build(self) -> list[InitStep]:
            return []

    builder.add_core_models([AlgorithmModel()])
    inits = builder.build()
    assert inits[0].name == "models.algorithmmodel"


def test_builder_add_optional_models(
    builder: InitializerBuilder, mock_optional_model: Any
) -> None:
    """add_optional_models adds run_build() InitSteps unchanged."""
    builder.add_optional_models([mock_optional_model])
    inits = builder.build()
    assert len(inits) >= 1
    assert any(li.name == "optional_field" and li.category == "fields" for li in inits)


def test_builder_add_and_extend(builder: InitializerBuilder) -> None:
    """add() and extend() store InitStep objects directly."""
    a = InitStep("a", initializer=lambda _ctx: 1)
    b = InitStep("b", initializer=lambda _ctx: 2)
    c = InitStep("c", initializer=lambda _ctx: 3)

    builder.add(a).extend([b, c])
    inits = builder.build()

    assert len(inits) == 3
    assert inits[0] is a


def test_builder_add_preserves_explicit_category(builder: InitializerBuilder) -> None:
    """Builder.add stores explicit category as provided by InitStep."""
    li = InitStep("custom", initializer=lambda _ctx: 1, category="resource")
    builder.add(li)
    assert builder.build()[0].category == "resource"


def test_builds_init_steps_protocol_matches_run_build(builder, mock_core_model):
    """`BuildsInitSteps` Protocol matches objects exposing run_build()."""
    from neofoam.framework.initialization.helpers import BuildsInitSteps

    class NoRunBuild:
        pass

    assert isinstance(mock_core_model, BuildsInitSteps)
    assert not isinstance(NoRunBuild(), BuildsInitSteps)


def test_lazy_accepts_replaces() -> None:
    """``lazy`` threads the replacement target onto the step."""
    step = lazy("mesh", lambda _ctx: 1, replaces=["mesh"])
    assert step.replaces == ["mesh"]


def test_field_accepts_replaces() -> None:
    """``field`` threads the replacement target onto the prefixed step."""
    step = field("U", lambda _ctx: 1, depends_on=["mesh"], replaces=["U"])
    assert step.name == "fields.U"
    assert step.replaces == ["U"]


def test_lazy_replaces_defaults_empty() -> None:
    """``lazy`` without a replacement target supersedes nothing."""
    assert lazy("mesh", lambda _ctx: 1).replaces == []


def test_build_drops_replaced_default_step() -> None:
    """A replacer named for a default supersedes that default step."""
    builder = InitializerBuilder()
    builder.add(lazy("mesh", lambda _ctx: "disk"))
    builder.add(lazy("mesh", lambda _ctx: "generated", replaces=["mesh"]))
    steps = builder.build()
    mesh_steps = [s for s in steps if s.name == "mesh"]
    assert len(mesh_steps) == 1
    assert mesh_steps[0].initializer({}) == "generated"


def test_build_keeps_dependent_orderable_after_replace() -> None:
    """Dependents of a replaced name still resolve against the carrier."""
    builder = InitializerBuilder()
    builder.add(lazy("mesh", lambda _ctx: "disk"))
    builder.add(lazy("mesh", lambda _ctx: "generated", replaces=["mesh"]))
    builder.add(field("U", lambda ctx: ctx["mesh"], depends_on=["mesh"]))
    names = [s.name for s in builder.build()]
    assert names.count("mesh") == 1
    assert "fields.U" in names


def test_build_keeps_all_when_no_replaces() -> None:
    """Without any replacement declaration every step survives build."""
    builder = InitializerBuilder()
    builder.add(lazy("a", lambda _ctx: 1))
    builder.add(lazy("b", lambda _ctx: 2))
    assert {s.name for s in builder.build()} == {"a", "b"}
