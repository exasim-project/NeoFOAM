# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Gradient scheme models."""

from typing import Annotated, Any, Literal, Union

from pydantic import BeforeValidator, Discriminator

from ._variant import OPENFOAM_CONTEXT, SchemeVariant
from .interpolation import InterpolationScheme

# -- Variants ----------------------------------------------------------------


class GaussGrad(SchemeVariant):
    type: Literal["Gauss"] = "Gauss"
    interpolation: InterpolationScheme

    def openfoam_str(self) -> str:
        interp = self.interpolation.model_dump(mode="python", context=OPENFOAM_CONTEXT)
        return f"Gauss {interp}"


class LeastSquaresGrad(SchemeVariant):
    type: Literal["pointCellsLeastSquares"] = "pointCellsLeastSquares"

    def openfoam_str(self) -> str:
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
