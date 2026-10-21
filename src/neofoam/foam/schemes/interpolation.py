# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Interpolation scheme models for face-value reconstruction."""

from typing import Annotated, Any, Literal, Union

from pydantic import BeforeValidator, Discriminator, Field

from ._variant import SchemeVariant

# -- Variants ----------------------------------------------------------------


class Linear(SchemeVariant):
    type: Literal["linear"] = "linear"

    def openfoam_str(self) -> str:
        return self.type


class Upwind(SchemeVariant):
    type: Literal["upwind"] = "upwind"

    def openfoam_str(self) -> str:
        return self.type


class LinearUpwind(SchemeVariant):
    type: Literal["linearUpwind"] = "linearUpwind"
    grad_field: str

    def openfoam_str(self) -> str:
        return f"linearUpwind {self.grad_field}"


class LimitedLinear(SchemeVariant):
    type: Literal["limitedLinear"] = "limitedLinear"
    coefficient: float = Field(ge=0, le=1)

    def openfoam_str(self) -> str:
        return f"limitedLinear {self.coefficient:g}"


class VanLeer(SchemeVariant):
    type: Literal["vanLeer"] = "vanLeer"

    def openfoam_str(self) -> str:
        return self.type


class Minmod(SchemeVariant):
    type: Literal["Minmod"] = "Minmod"

    def openfoam_str(self) -> str:
        return self.type


class SuperBee(SchemeVariant):
    type: Literal["SuperBee"] = "SuperBee"

    def openfoam_str(self) -> str:
        return self.type


class MUSCL(SchemeVariant):
    type: Literal["MUSCL"] = "MUSCL"

    def openfoam_str(self) -> str:
        return self.type


class QUICK(SchemeVariant):
    type: Literal["QUICK"] = "QUICK"

    def openfoam_str(self) -> str:
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
