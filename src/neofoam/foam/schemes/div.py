# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Divergence scheme models."""

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, BeforeValidator, Discriminator, model_serializer

from .interpolation import InterpolationScheme


# -- Variants ----------------------------------------------------------------


class NoneDiv(BaseModel):
    type: Literal["none"] = "none"

    @model_serializer
    def serialize(self) -> str:
        return self.type


class GaussDiv(BaseModel):
    type: Literal["Gauss"] = "Gauss"
    interpolation: InterpolationScheme

    @model_serializer
    def serialize(self) -> str:
        interp = self.interpolation.model_dump(mode="python")
        return f"Gauss {interp}"


class BoundedGaussDiv(BaseModel):
    type: Literal["bounded"] = "bounded"
    interpolation: InterpolationScheme

    @model_serializer
    def serialize(self) -> str:
        interp = self.interpolation.model_dump(mode="python")
        return f"bounded Gauss {interp}"


# -- Parser + Union -----------------------------------------------------------


def _parse_div(v: Any) -> Any:
    if not isinstance(v, str):
        return v
    if v == "none":
        return {"type": "none"}
    rest = v
    if rest.startswith("bounded "):
        rest = rest[len("bounded ") :]
        tokens = rest.split(maxsplit=1)
        result: dict[str, Any] = {"type": "bounded"}
        if len(tokens) > 1:
            result["interpolation"] = tokens[1]
        return result
    tokens = rest.split(maxsplit=1)
    result = {"type": tokens[0]}
    if len(tokens) > 1:
        result["interpolation"] = tokens[1]
    return result


DivScheme = Annotated[
    Union[NoneDiv, GaussDiv, BoundedGaussDiv],
    BeforeValidator(_parse_div),
    Discriminator("type"),
]
