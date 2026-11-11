# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""A coupled side: its exchange models hold Array fields only; its coordinates are floats."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import BaseModel

from neofoam.framework.coupling import Array
from neofoam.framework.coupling.data import ExchangeSide


class WallTemperature(BaseModel):
    T: Array


class _Plain(BaseModel):
    q: Array
    h: float


class _Listed(BaseModel):
    weights: list[float]


@pytest.mark.parametrize(("model", "field"), [(_Plain, "h"), (_Listed, "weights")])
def test_an_exchange_model_holds_array_fields_only(model: type[BaseModel], field: str) -> None:
    with pytest.raises(
        TypeError,
        match=f"the field '{field}' of the exchange model {model.__name__} is not Array",
    ):
        ExchangeSide(
            "solid",
            "right",
            "T",
            sends=model,
            receives=WallTemperature,
            send=lambda: None,
            receive=print,
        )


def test_a_side_keeps_its_coordinates_as_a_float_array() -> None:
    side = ExchangeSide(
        "solid",
        "right",
        "T",
        sends=WallTemperature,
        receives=WallTemperature,
        send=lambda: None,
        receive=print,
        coordinates=[[0, 0, 0], [1, 0, 0]],
    )

    assert isinstance(side.coordinates, np.ndarray) and side.coordinates.dtype == float
    assert side.points == 2
