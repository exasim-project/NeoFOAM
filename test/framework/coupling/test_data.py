# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Data models hold Array (or Scalar) fields only; a port merges the streams of its instances."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from neofoam.framework.coupling import Array, Cells, FaceData, Global, Patch, Scalar, ValueData
from neofoam.framework.coupling.data import CoupledPort, Stream, check_model, key


class WallTemperature(FaceData):
    T: Array


class _Plain(FaceData):
    q: Array
    h: float


class _Listed(FaceData):
    weights: list[float]


class _Value(ValueData):
    q: Scalar


class _Bare(BaseModel):
    q: Array


@pytest.mark.parametrize(
    ("model", "error"),
    [
        (_Plain, "_Plain.h is not Array: a FaceData holds Array fields only"),
        (_Listed, "_Listed.weights is not Array"),
        (_Bare, "declares <class .*_Bare'>, which is no FaceData, CellData, ValueData or Probe"),
    ],
    ids=["float", "list", "no data base"],
)
def test_a_data_model_holds_fields_of_its_kind_only(model: type, error: str) -> None:
    with pytest.raises(TypeError, match=error):
        check_model(model, "A.send (@coupled.sends)")


def test_a_value_model_holds_scalars() -> None:
    assert check_model(_Value, "A.send") is _Value


class _Pair(FaceData):
    h: Array
    T_sink: Array


def test_a_model_s_key_is_its_sorted_field_names() -> None:
    assert (key(WallTemperature), key(_Pair)) == ("T", "T_sink,h")


def _port(location: object, *sends: type) -> CoupledPort:
    return CoupledPort("a", location, tuple(Stream(model, print) for model in sends))


def test_a_port_merges_the_streams_of_another_instance() -> None:
    merged = _port(Patch("top"), WallTemperature).merged(_port(Patch("top"), _Value))

    assert [stream.model.__name__ for stream in merged.sends] == ["WallTemperature", "_Value"]


def test_a_port_sends_a_data_field_once() -> None:
    with pytest.raises(ValueError, match=r"two @coupled.sends of T on a.top"):
        _port(Patch("top"), WallTemperature).merged(_port(Patch("top"), WallTemperature))


def test_a_port_name_names_one_location() -> None:
    with pytest.raises(ValueError, match=r"a.cells is bound to Cells\(name='cells'\) and Global"):
        _port(Cells(), WallTemperature).merged(_port(Global("cells"), _Value))
