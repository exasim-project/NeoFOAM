# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Unit tests for InitResult, the immutable per-step initialization output."""

import dataclasses

import pytest

from neofoam.framework.initialization.execution.init_result import InitResult


def test_init_result_is_immutable():
    result = InitResult(name="fields.U", category="fields", value="vel")

    with pytest.raises(dataclasses.FrozenInstanceError):
        result.value = "other"  # type: ignore[misc]


def test_init_result_carries_name_category_value():
    result = InitResult(name="models.algo", category="models", value=42)

    assert result.name == "models.algo"
    assert result.category == "models"
    assert result.value == 42
