# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Time discretization (ddt) scheme models."""

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, BeforeValidator, Discriminator, Field, model_serializer


# -- Variants ----------------------------------------------------------------


class Euler(BaseModel):
    type: Literal["Euler"] = "Euler"

    @model_serializer
    def serialize(self) -> str:
        return self.type


class Backward(BaseModel):
    type: Literal["backward"] = "backward"

    @model_serializer
    def serialize(self) -> str:
        return self.type


class SteadyState(BaseModel):
    type: Literal["steadyState"] = "steadyState"

    @model_serializer
    def serialize(self) -> str:
        return self.type


class LocalEuler(BaseModel):
    type: Literal["localEuler"] = "localEuler"

    @model_serializer
    def serialize(self) -> str:
        return self.type


class CrankNicolson(BaseModel):
    type: Literal["CrankNicolson"] = "CrankNicolson"
    coefficient: float = Field(ge=0, le=1)

    @model_serializer
    def serialize(self) -> str:
        return f"CrankNicolson {self.coefficient:g}"


# -- Parser + Union -----------------------------------------------------------


def _parse_ddt(v: Any) -> Any:
    if not isinstance(v, str):
        return v
    tokens = v.split()
    result: dict[str, Any] = {"type": tokens[0]}
    if len(tokens) > 1 and tokens[0] == "CrankNicolson":
        result["coefficient"] = float(tokens[1])
    return result


DdtScheme = Annotated[
    Union[Euler, Backward, SteadyState, LocalEuler, CrankNicolson],
    BeforeValidator(_parse_ddt),
    Discriminator("type"),
]
