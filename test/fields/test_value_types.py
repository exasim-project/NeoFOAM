# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for ``neofoam.fields.value_types``: uniform literal round-trips."""

from __future__ import annotations

import pytest

from neofoam.fields.value_types import (
    Scalar,
    Vector,
    default_uniform,
    parse_uniform,
    zero_uniform,
)


# -- scalar -----------------------------------------------------------


def test_scalar_default_from_float() -> None:
    assert default_uniform(Scalar, 0.0) == "uniform 0.0"
    assert default_uniform(Scalar, 1.5) == "uniform 1.5"


def test_scalar_default_passes_string_through() -> None:
    """An already-formatted literal is returned untouched."""
    assert default_uniform(Scalar, "uniform 300") == "uniform 300"


def test_scalar_parse() -> None:
    assert parse_uniform(Scalar, "uniform 0") == 0.0
    assert parse_uniform(Scalar, "uniform 300") == 300.0
    assert parse_uniform(Scalar, "uniform 1.5e-3") == pytest.approx(1.5e-3)


def test_scalar_parse_rejects_vector_literal() -> None:
    with pytest.raises(ValueError):
        parse_uniform(Scalar, "uniform (0 0 0)")


def test_scalar_round_trip() -> None:
    assert parse_uniform(Scalar, default_uniform(Scalar, 42.0)) == 42.0


# -- vector -----------------------------------------------------------


def test_vector_default_from_list() -> None:
    assert default_uniform(Vector, [1.0, 2.0, 3.0]) == "uniform (1.0 2.0 3.0)"


def test_vector_default_from_tuple() -> None:
    assert default_uniform(Vector, (0.0, 0.0, 0.0)) == "uniform (0.0 0.0 0.0)"


def test_vector_default_rejects_bad_shape() -> None:
    with pytest.raises(TypeError):
        default_uniform(Vector, [1.0, 2.0])  # only two components


def test_vector_parse() -> None:
    assert parse_uniform(Vector, "uniform (0 0 0)") == (0.0, 0.0, 0.0)
    assert parse_uniform(Vector, "uniform (1 -2 3.5)") == (1.0, -2.0, 3.5)


def test_vector_parse_rejects_scalar_literal() -> None:
    with pytest.raises(ValueError):
        parse_uniform(Vector, "uniform 0")


def test_vector_round_trip() -> None:
    assert parse_uniform(Vector, default_uniform(Vector, [1.0, 2.0, 3.0])) == (
        1.0,
        2.0,
        3.0,
    )


# -- zero / unsupported ----------------------------------------------


def test_zero_uniform() -> None:
    assert zero_uniform(Scalar) == "uniform 0"
    assert zero_uniform(Vector) == "uniform (0 0 0)"


def test_default_uniform_rejects_unknown_value_type() -> None:
    class _Bogus:
        pass

    with pytest.raises(TypeError):
        default_uniform(_Bogus, 0.0)


def test_parse_uniform_rejects_unknown_value_type() -> None:
    class _Bogus:
        pass

    with pytest.raises(TypeError):
        parse_uniform(_Bogus, "uniform 0")
