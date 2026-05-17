# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

"""Unit tests for initialization helpers."""

import pytest

from neofoam.framework.initialization.helpers import (
    field,
    lazy,
    model,
    operator,
)
from neofoam.framework.initialization.init_step import InitStep


# --- Helper function tests (parametrized) ---


@pytest.mark.parametrize(
    "helper,prefix,category",
    [
        (field, "fields.", "fields"),
        (model, "models.", "models"),
        (operator, "operators.", "operators"),
        (lazy, "", "resource"),
    ],
)
def test_helper_naming(helper, prefix, category):
    """All helpers create InitStep with correct name prefix and category."""
    result = helper("X", create=lambda _ctx: 1)
    assert isinstance(result, InitStep)
    assert result.name == f"{prefix}X"
    assert result.category == category
    assert result.depends_on == []


@pytest.mark.parametrize("helper", [field, model, operator, lazy])
def test_helper_with_deps(helper):
    """All helpers pass through custom depends_on."""
    result = helper("X", create=lambda _ctx: 1, depends_on=["a", "b"])
    assert result.depends_on == ["a", "b"]


def test_helper_execute():
    """Helpers produce executable InitStep objects."""
    assert field("U", create=lambda _ctx: "velocity").execute({}) == "velocity"
    assert lazy("mesh", create=lambda _ctx: "mesh_obj").execute({}) == "mesh_obj"


# --- InitializerBuilder tests ---


@pytest.mark.parametrize(
    "method,expected_prefix,kwargs",
    [
        ("add_field", "fields.", {"depends_on": ["mesh"], "value": 42}),
        ("add_model", "models.", {"value": "instance"}),
    ],
)
def test_builder_add_methods(builder, method, expected_prefix, kwargs):
    """Builder add_field/add_model create correctly prefixed InitStep."""
    getattr(builder, method)("X", **kwargs)
    inits = builder.build()
    assert len(inits) == 1
    assert inits[0].name == f"{expected_prefix}X"


def test_builder_add_resource(builder):
    """add_resource creates an unprefixed InitStep."""
    builder.add_resource("mesh", "mock_mesh")
    inits = builder.build()
    assert len(inits) == 1
    assert inits[0].name == "mesh"
    assert inits[0].execute({}) == "mock_mesh"


def test_builder_add_field_callable(builder):
    """add_field with callable value passes the callable through."""
    builder.add_field("p", depends_on=["mesh"], value=lambda _ctx: "computed")
    assert builder.build()[0].execute({}) == "computed"


def test_builder_add_model_callable(builder):
    """add_model with callable value passes the callable through."""
    builder.add_model("turb", value=lambda _ctx: "turb_inst")
    assert builder.build()[0].execute({}) == "turb_inst"


def test_builder_chaining(builder):
    """All builder methods return self for chaining."""
    result = (
        builder.add_resource("mesh", "m")
        .add_field("U", depends_on=["mesh"], value=1)
        .add_model("algo", value=2)
    )
    assert result is builder
    assert len(builder.build()) == 3


def test_builder_add_core_models(builder, mock_core_model):
    """add_core_models adds model + InitSteps from run_build() unchanged."""
    builder.add_core_models([("algorithm", mock_core_model)])
    inits = builder.build()

    assert len(inits) >= 3
    model_inits = [li for li in inits if li.name == "models.algorithm"]
    assert len(model_inits) == 1
    assert any(li.name == "test_field" and li.category == "fields" for li in inits)
    assert any(li.name == "test_op" and li.category == "operators" for li in inits)


def test_builder_add_core_models_without_name(builder):
    """add_core_models uses lowercase class name when no tuple name given."""

    class AlgorithmModel:
        def run_build(self):
            return []

    builder.add_core_models([AlgorithmModel()])
    inits = builder.build()
    assert inits[0].name == "models.algorithmmodel"


def test_builder_add_optional_models(builder, mock_optional_model):
    """add_optional_models adds run_build() InitSteps unchanged."""
    builder.add_optional_models([mock_optional_model])
    inits = builder.build()
    assert len(inits) >= 1
    assert any(li.name == "optional_field" and li.category == "fields" for li in inits)


def test_builder_add_and_extend(builder):
    """add() and extend() store InitStep objects directly."""
    a = InitStep("a", initializer=lambda _ctx: 1)
    b = InitStep("b", initializer=lambda _ctx: 2)
    c = InitStep("c", initializer=lambda _ctx: 3)

    builder.add(a).extend([b, c])
    inits = builder.build()

    assert len(inits) == 3
    assert inits[0] is a


def test_builder_add_preserves_explicit_category(builder):
    """Builder.add stores explicit category as provided by InitStep."""
    li = InitStep("custom", initializer=lambda _ctx: 1, category="resource")
    builder.add(li)
    assert builder.build()[0].category == "resource"
