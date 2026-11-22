# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Surface-normal gradient scheme models."""

from typing import Annotated, Any, Literal, Union

from pydantic import BeforeValidator, Discriminator, Field

from ._variant import SchemeVariant

# -- Variants ----------------------------------------------------------------


class Corrected(SchemeVariant):
    type: Literal["corrected"] = "corrected"

    def openfoam_str(self) -> str:
        return self.type


class Uncorrected(SchemeVariant):
    type: Literal["uncorrected"] = "uncorrected"

    def openfoam_str(self) -> str:
        return self.type


class Orthogonal(SchemeVariant):
    type: Literal["orthogonal"] = "orthogonal"

    def openfoam_str(self) -> str:
        return self.type


class LimitedSnGrad(SchemeVariant):
    type: Literal["limited"] = "limited"
    coefficient: float = Field(gt=0, le=1)

    def openfoam_str(self) -> str:
        return f"limited {self.coefficient:g}"


# -- Parser + Union -----------------------------------------------------------


def _parse_sn_grad(v: Any) -> Any:
    if not isinstance(v, str):
        return v
    tokens = v.split()
    result: dict[str, Any] = {"type": tokens[0]}
    if len(tokens) > 1 and tokens[0] == "limited":
        # OpenFOAM's limitedSnGrad takes an optional sub-scheme token before the
        # coefficient, so "limited 0.33" and "limited corrected 0.33" are the same
        # scheme (the terse form is what serialize writes back). Only "corrected"
        # is a valid sub-scheme -- NeoN rejects any other, so we must too.
        arguments = tokens[1:]
        if len(arguments) > 1:
            if arguments[0] != "corrected":
                raise ValueError(f"unsupported limited snGrad sub-scheme: {arguments[0]!r}")
            arguments = arguments[1:]
        result["coefficient"] = float(arguments[0])
    return result


SnGradScheme = Annotated[
    Union[Corrected, Uncorrected, Orthogonal, LimitedSnGrad],
    BeforeValidator(_parse_sn_grad),
    Discriminator("type"),
]
