# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for the JSON-schema form helpers (``neofoam.io.pydantic_schema``).

Pure pydantic/dict manipulation — no marimo, no OpenFOAM. Uses small local
models that reproduce the shapes the case notebooks render: a discriminated
union (``oneOf``/``anyOf`` + a ``const`` discriminator), a nested union, and a
patch dict via ``additionalProperties``.
"""

from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import BaseModel, Field

from neofoam.io.pydantic_schema import default_values, rjsf_uischema, slice_schema


class _Euler(BaseModel):
    type: Literal["Euler"] = "Euler"


class _CrankNicolson(BaseModel):
    type: Literal["CrankNicolson"] = "CrankNicolson"
    coefficient: float = 0.9


class _Ddt(BaseModel):
    ddt: Union[_Euler, _CrankNicolson] = Field(discriminator="type")


def test_default_values_uses_construct_and_alias() -> None:
    class M(BaseModel):
        a: int = 3
        b: str = "x"

    assert default_values(M) == {"a": 3, "b": "x"}


def test_default_values_never_raises() -> None:
    class Req(BaseModel):
        needed: int  # required, no default

    # model_construct skips validation, so this yields a (partial) dict, not a raise.
    assert isinstance(default_values(Req), dict)


def test_default_values_prefers_form_defaults_scaffold() -> None:
    # A config whose fields are required-but-defaultless (model_construct → {}) can
    # supply a ready-to-edit scaffold via form_defaults; default_values returns it.
    from neofoam.io.base import BaseConfig

    class _Scaffolded(BaseConfig):
        needed: int  # required, no default

        @classmethod
        def form_defaults(cls) -> dict[str, object]:
            return {"needed": 42}

    assert default_values(_Scaffolded) == {"needed": 42}


def test_default_values_falls_back_when_no_scaffold() -> None:
    # BaseConfig.form_defaults returns None by default → the model_construct path.
    from neofoam.io.base import BaseConfig

    class _Plain(BaseConfig):
        a: int = 3

    assert default_values(_Plain) == {"a": 3}


def test_slice_schema_keeps_named_props_and_filters_required() -> None:
    schema = {
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "string"}},
        "required": ["a", "b"],
        "$defs": {"X": {"type": "object"}},
    }
    sliced = slice_schema(schema, {"a"})
    assert set(sliced["properties"]) == {"a"}
    assert sliced["required"] == ["a"]
    assert sliced["$defs"] == {"X": {"type": "object"}}  # preserved
    # original untouched
    assert set(schema["properties"]) == {"a", "b"}


def test_slice_schema_drops_required_when_empty() -> None:
    schema = {"properties": {"a": {}, "b": {}}, "required": ["b"]}
    sliced = slice_schema(schema, {"a"})
    assert "required" not in sliced


def test_rjsf_uischema_hides_const_discriminator() -> None:
    ui = rjsf_uischema(_Ddt.model_json_schema())
    # The union field tidies its branches: the const ``type`` is hidden, and the
    # extra ``coefficient`` field carries no widget override.
    assert ui["ddt"]["type"] == {"ui:widget": "hidden"}
    assert "ui:options" not in ui["ddt"]  # no label:false (keeps the key label)


def test_rjsf_uischema_handles_anyof_additionalproperties() -> None:
    class _FixedValue(BaseModel):
        type: Literal["fixedValue"] = "fixedValue"
        value: float = 0.0

    class _NoSlip(BaseModel):
        type: Literal["noSlip"] = "noSlip"

    class _Field(BaseModel):
        boundaryField: dict[str, Union[_FixedValue, _NoSlip]] = {}

    ui = rjsf_uischema(_Field.model_json_schema())
    # The BC union is reached through ``additionalProperties``; its const type
    # is hidden there.
    assert ui["boundaryField"]["additionalProperties"]["type"] == {
        "ui:widget": "hidden"
    }


def test_rjsf_uischema_nested_union() -> None:
    class _Linear(BaseModel):
        type: Literal["linear"] = "linear"

    class _Gauss(BaseModel):
        type: Literal["Gauss"] = "Gauss"
        interpolation: Union[_Linear, "_Linear"] = _Linear()

    class _Grad(BaseModel):
        grad: Union[_Gauss, _Linear] = Field(discriminator="type")

    ui = rjsf_uischema(_Grad.model_json_schema())
    assert ui["grad"]["type"] == {"ui:widget": "hidden"}
    # nested interpolation union is also tidied
    assert ui["grad"]["interpolation"]["type"] == {"ui:widget": "hidden"}


def test_rjsf_uischema_leaves_plain_object_alone() -> None:
    class Plain(BaseModel):
        a: int = 1
        b: Optional[str] = None

    # No unions / consts → empty uiSchema (nothing to tidy).
    assert rjsf_uischema(Plain.model_json_schema()) == {}
