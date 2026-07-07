# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Surface-normal gradient scheme models."""

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, BeforeValidator, Discriminator, Field, model_serializer


# -- Variants ----------------------------------------------------------------


class Corrected(BaseModel):
    type: Literal["corrected"] = "corrected"

    @model_serializer
    def serialize(self) -> str:
        return self.type


class Uncorrected(BaseModel):
    type: Literal["uncorrected"] = "uncorrected"

    @model_serializer
    def serialize(self) -> str:
        return self.type


class Orthogonal(BaseModel):
    type: Literal["orthogonal"] = "orthogonal"

    @model_serializer
    def serialize(self) -> str:
        return self.type


class LimitedSnGrad(BaseModel):
    type: Literal["limited"] = "limited"
    coefficient: float = Field(gt=0, le=1)

    @model_serializer
    def serialize(self) -> str:
        return f"limited {self.coefficient:g}"


# -- Parser + Union -----------------------------------------------------------


def _parse_sn_grad(v: Any) -> Any:
    if not isinstance(v, str):
        return v
    tokens = v.split()
    result: dict[str, Any] = {"type": tokens[0]}
    if len(tokens) > 1 and tokens[0] == "limited":
        result["coefficient"] = float(tokens[1])
    return result


SnGradScheme = Annotated[
    Union[Corrected, Uncorrected, Orthogonal, LimitedSnGrad],
    BeforeValidator(_parse_sn_grad),
    Discriminator("type"),
]
