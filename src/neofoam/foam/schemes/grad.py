# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Gradient scheme models."""

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, BeforeValidator, Discriminator, model_serializer

from .interpolation import InterpolationScheme


# -- Variants ----------------------------------------------------------------


class GaussGrad(BaseModel):
    type: Literal["Gauss"] = "Gauss"
    interpolation: InterpolationScheme

    @model_serializer
    def serialize(self) -> str:
        interp = self.interpolation.model_dump(mode="python")
        return f"Gauss {interp}"


class LeastSquaresGrad(BaseModel):
    type: Literal["pointCellsLeastSquares"] = "pointCellsLeastSquares"

    @model_serializer
    def serialize(self) -> str:
        return self.type


# -- Parser + Union -----------------------------------------------------------


def _parse_grad(v: Any) -> Any:
    if not isinstance(v, str):
        return v
    tokens = v.split(maxsplit=1)
    result: dict[str, Any] = {"type": tokens[0]}
    if len(tokens) > 1:
        result["interpolation"] = tokens[1]
    return result


GradScheme = Annotated[
    Union[GaussGrad, LeastSquaresGrad],
    BeforeValidator(_parse_grad),
    Discriminator("type"),
]
