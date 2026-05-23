# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Interpolation scheme models for face-value reconstruction."""

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, BeforeValidator, Discriminator, Field, model_serializer


# -- Variants ----------------------------------------------------------------


class Linear(BaseModel):
    type: Literal["linear"] = "linear"

    @model_serializer
    def serialize(self) -> str:
        return self.type


class Upwind(BaseModel):
    type: Literal["upwind"] = "upwind"

    @model_serializer
    def serialize(self) -> str:
        return self.type


class LinearUpwind(BaseModel):
    type: Literal["linearUpwind"] = "linearUpwind"
    grad_field: str

    @model_serializer
    def serialize(self) -> str:
        return f"linearUpwind {self.grad_field}"


class LimitedLinear(BaseModel):
    type: Literal["limitedLinear"] = "limitedLinear"
    coefficient: float = Field(ge=0, le=1)

    @model_serializer
    def serialize(self) -> str:
        return f"limitedLinear {self.coefficient:g}"


class VanLeer(BaseModel):
    type: Literal["vanLeer"] = "vanLeer"

    @model_serializer
    def serialize(self) -> str:
        return self.type


class Minmod(BaseModel):
    type: Literal["Minmod"] = "Minmod"

    @model_serializer
    def serialize(self) -> str:
        return self.type


class SuperBee(BaseModel):
    type: Literal["SuperBee"] = "SuperBee"

    @model_serializer
    def serialize(self) -> str:
        return self.type


class MUSCL(BaseModel):
    type: Literal["MUSCL"] = "MUSCL"

    @model_serializer
    def serialize(self) -> str:
        return self.type


class QUICK(BaseModel):
    type: Literal["QUICK"] = "QUICK"

    @model_serializer
    def serialize(self) -> str:
        return self.type


# -- Parser + Union -----------------------------------------------------------

_INTERP_ARG_FIELD = {
    "linearUpwind": "grad_field",
    "limitedLinear": "coefficient",
}


def _parse_interpolation(v: Any) -> Any:
    if not isinstance(v, str):
        return v
    tokens = v.split(maxsplit=1)
    result: dict[str, Any] = {"type": tokens[0]}
    if len(tokens) > 1:
        field_name = _INTERP_ARG_FIELD.get(tokens[0], "argument")
        raw = tokens[1]
        result[field_name] = float(raw) if field_name == "coefficient" else raw
    return result


InterpolationScheme = Annotated[
    Union[
        Linear,
        Upwind,
        LinearUpwind,
        LimitedLinear,
        VanLeer,
        Minmod,
        SuperBee,
        MUSCL,
        QUICK,
    ],
    BeforeValidator(_parse_interpolation),
    Discriminator("type"),
]
