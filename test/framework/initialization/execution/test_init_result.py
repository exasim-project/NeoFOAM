# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

from neofoam.framework.initialization.execution.init_result import InitResult


def test_init_result_is_immutable():
    result = InitResult(name="fields.U", category="fields", value="vel")

    try:
        result.value = "other"  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("InitResult should be frozen")


def test_init_result_carries_name_category_value():
    result = InitResult(name="models.algo", category="models", value=42)

    assert result.name == "models.algo"
    assert result.category == "models"
    assert result.value == 42
