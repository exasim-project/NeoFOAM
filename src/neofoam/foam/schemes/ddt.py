# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Time discretization (ddt) scheme models."""

from typing import Annotated, Any, Literal, Union

from pydantic import BeforeValidator, Discriminator, Field

from ._variant import SchemeVariant

# -- Variants ----------------------------------------------------------------


class Euler(SchemeVariant):
    type: Literal["Euler"] = "Euler"

    def openfoam_str(self) -> str:
        return self.type


class Backward(SchemeVariant):
    type: Literal["backward"] = "backward"

    def openfoam_str(self) -> str:
        return self.type


class SteadyState(SchemeVariant):
    type: Literal["steadyState"] = "steadyState"

    def openfoam_str(self) -> str:
        return self.type


class LocalEuler(SchemeVariant):
    type: Literal["localEuler"] = "localEuler"

    def openfoam_str(self) -> str:
        return self.type


class CrankNicolson(SchemeVariant):
    type: Literal["CrankNicolson"] = "CrankNicolson"
    coefficient: float = Field(ge=0, le=1)

    def openfoam_str(self) -> str:
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
