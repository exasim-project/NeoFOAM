# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Gradient scheme models."""

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, BeforeValidator, Discriminator, Field, model_serializer

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


class CellLimitedGrad(BaseModel):
    """``cellLimited <inner grad scheme> <k>`` — minmod slope limiter on an inner scheme."""

    type: Literal["cellLimited"] = "cellLimited"
    inner_scheme: "GradScheme"
    # k is the limiter strength, 1 being full limiting (OpenFOAM widens the
    # admissible cell range by (1/k - 1)*(max - min)).
    coefficient: float = Field(ge=0, le=1)

    @model_serializer
    def serialize(self) -> str:
        inner = self.inner_scheme.model_dump(mode="python")
        return f"cellLimited {inner} {self.coefficient:g}"


# -- Parser + Union -----------------------------------------------------------


def _parse_cell_limited_grad(v: str) -> dict[str, Any]:
    """``cellLimited Gauss linear 1`` — inner scheme in the middle, coefficient last."""
    tokens = v.split()
    if len(tokens) < 3:
        raise ValueError(f"cellLimited grad needs an inner scheme and a coefficient: {v!r}")
    return {
        "type": "cellLimited",
        "inner_scheme": " ".join(tokens[1:-1]),
        "coefficient": float(tokens[-1]),
    }


def _parse_grad(v: Any) -> Any:
    if not isinstance(v, str):
        return v
    tokens = v.split(maxsplit=1)
    if tokens[0] == "cellLimited":
        return _parse_cell_limited_grad(v)
    result: dict[str, Any] = {"type": tokens[0]}
    if len(tokens) > 1:
        result["interpolation"] = tokens[1]
    return result


GradScheme = Annotated[
    Union[GaussGrad, LeastSquaresGrad, CellLimitedGrad],
    BeforeValidator(_parse_grad),
    Discriminator("type"),
]

# CellLimitedGrad nests a GradScheme, which is only defined above — resolve it now.
CellLimitedGrad.model_rebuild()
