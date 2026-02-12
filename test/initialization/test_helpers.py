# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

"""Unit tests for initialization helpers."""

import pytest

from neofoam.framework.initialization.helpers import (
    field,
    operator,
    lazy,
    model,
)
from neofoam.framework.initialization.lazy_init import LazyInit


# --- Helper function tests (parametrized) ---


@pytest.mark.parametrize(
    "helper,prefix,category",
    [
        (field, "fields.", "fields"),
        (operator, "operators.", "operators"),
        (model, "models.", "models"),
        (lazy, "", None),
    ],
)
def test_helper_naming(helper, prefix, category):
    """All helpers create LazyInit with correct name prefix and category."""
    result = helper("X", create=lambda: 1)
    assert isinstance(result, LazyInit)
    assert result.name == f"{prefix}X"
    assert result.category == category
    assert result.depends_on == []


@pytest.mark.parametrize("helper", [field, operator, model, lazy])
def test_helper_with_deps(helper):
    """All helpers pass through custom depends_on."""
    result = helper("X", create=lambda: 1, depends_on=["a", "b"])
    assert result.depends_on == ["a", "b"]


def test_helper_execute():
    """Helpers produce executable LazyInit objects."""
    assert field("U", create=lambda: "velocity").execute() == "velocity"
    assert lazy("mesh", create=lambda: "mesh_obj").execute() == "mesh_obj"


# --- InitializerBuilder tests ---


@pytest.mark.parametrize(
    "method,expected_prefix,kwargs",
    [
        ("add_field", "fields.", {"depends_on": ["mesh"], "value": 42}),
        ("add_model", "models.", {"value": "instance"}),
        ("add_operator", "operators.", {"depends_on": ["fields.U"], "value": "op"}),
    ],
)
def test_builder_add_methods(builder, method, expected_prefix, kwargs):
    """Builder add_field/add_model/add_operator create correctly prefixed LazyInit."""
    getattr(builder, method)("X", **kwargs)
    inits = builder.build()
    assert len(inits) == 1
    assert inits[0].name == f"{expected_prefix}X"


def test_builder_add_resource(builder):
    """add_resource creates an unprefixed LazyInit."""
    builder.add_resource("mesh", "mock_mesh")
    inits = builder.build()
    assert len(inits) == 1
    assert inits[0].name == "mesh"
    assert inits[0].execute() == "mock_mesh"


def test_builder_add_field_callable(builder):
    """add_field with callable value passes the callable through."""
    builder.add_field("p", depends_on=["mesh"], value=lambda: "computed")
    assert builder.build()[0].execute() == "computed"


def test_builder_add_model_callable(builder):
    """add_model with callable value passes the callable through."""
    builder.add_model("turb", value=lambda: "turb_inst")
    assert builder.build()[0].execute() == "turb_inst"


def test_builder_chaining(builder):
    """All builder methods return self for chaining."""
    result = (
        builder.add_resource("mesh", "m")
        .add_field("U", depends_on=["mesh"], value=1)
        .add_model("algo", value=2)
        .add_operator("mom", depends_on=[], value=3)
    )
    assert result is builder
    assert len(builder.build()) == 4


def test_builder_add_core_models(builder, mock_core_model):
    """add_core_models adds model + normalized LazyInits from run_build()."""
    builder.add_core_models([("algorithm", mock_core_model)])
    inits = builder.build()

    assert len(inits) >= 3
    model_inits = [li for li in inits if li.name == "models.algorithm"]
    assert len(model_inits) == 1


def test_builder_add_core_models_without_name(builder):
    """add_core_models uses lowercase class name when no tuple name given."""

    class AlgorithmModel:
        def run_build(self):
            return []

    builder.add_core_models([AlgorithmModel()])
    inits = builder.build()
    assert inits[0].name == "models.algorithmmodel"


def test_builder_add_optional_models(builder, mock_optional_model):
    """add_optional_models adds normalized LazyInits from run_build()."""
    builder.add_optional_models([mock_optional_model])
    assert len(builder.build()) >= 1


def test_builder_add_and_extend(builder):
    """add() and extend() store LazyInit objects directly."""
    a = LazyInit("a", initializer=lambda: 1)
    b = LazyInit("b", initializer=lambda: 2)
    c = LazyInit("c", initializer=lambda: 3)

    builder.add(a).extend([b, c])
    inits = builder.build()

    assert len(inits) == 3
    assert inits[0] is a


def test_builder_normalize_lazy_init(builder):
    """_normalize_lazy_init adds fields. prefix when missing."""
    li = LazyInit("U", initializer=lambda: "velocity")
    normalized = builder._normalize_lazy_init(li)
    assert normalized.name == "fields.U"


def test_builder_normalize_lazy_init_dict_unwrapping(builder):
    """Normalized initializers unwrap dict with 'value' key."""
    li = LazyInit("p", initializer=lambda: {"value": 42, "meta": "extra"})
    normalized = builder._normalize_lazy_init(li)
    assert normalized.execute(context={}) == 42


def test_builder_normalize_does_not_mutate(builder):
    """_normalize_lazy_init returns a new object; original is unchanged."""

    def orig_init():
        return "v"

    li = LazyInit("U", initializer=orig_init)
    normalized = builder._normalize_lazy_init(li)
    # Original must be untouched
    assert li.name == "U"
    assert li.initializer is orig_init
    # Normalized is a different object
    assert normalized is not li
    assert normalized.name == "fields.U"
