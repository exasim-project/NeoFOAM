# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Laplacian scheme models."""

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, BeforeValidator, Discriminator, model_serializer

from .interpolation import InterpolationScheme
from .sn_grad import SnGradScheme


# -- Variants ----------------------------------------------------------------


class NoneLaplacian(BaseModel):
    """``laplacianSchemes { default none; }`` — OpenFOAM's *no-default* sentinel."""

    type: Literal["none"] = "none"

    @model_serializer
    def serialize(self) -> str:
        return self.type


class GaussLaplacian(BaseModel):
    type: Literal["Gauss"] = "Gauss"
    interpolation: InterpolationScheme
    sn_grad: SnGradScheme

    @model_serializer
    def serialize(self) -> str:
        interp = self.interpolation.model_dump(mode="python")
        sn = self.sn_grad.model_dump(mode="python")
        return f"Gauss {interp} {sn}"


# -- Parser -------------------------------------------------------------------


def _parse_laplacian(v: Any) -> Any:
    if not isinstance(v, str):
        return v
    if v.strip() == "none":
        return {"type": "none"}
    tokens = v.split()
    result: dict[str, Any] = {"type": tokens[0]}
    if len(tokens) > 1:
        result["interpolation"] = tokens[1]
    if len(tokens) > 2:
        result["sn_grad"] = " ".join(tokens[2:])
    return result


LaplacianScheme = Annotated[
    Union[NoneLaplacian, GaussLaplacian],
    BeforeValidator(_parse_laplacian),
    Discriminator("type"),
]
