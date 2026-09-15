# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors
"""
Tests for the unified condition system.
"""

from neofoam.framework.conditions import Condition


def always_true() -> Condition:
    return Condition(lambda: True, "AlwaysTrue")


def always_false() -> Condition:
    return Condition(lambda: False, "AlwaysFalse")


def test_condition_evaluates_truthiness() -> None:
    assert always_true()()
    assert not always_false()()


def test_condition_and_is_false_when_one_operand_false() -> None:
    and_condition = always_true() & always_false()
    assert not and_condition()
    assert and_condition.name == "(AlwaysTrue & AlwaysFalse)"


def test_condition_or_is_true_when_one_operand_true() -> None:
    or_condition = always_true() | always_false()
    assert or_condition()
    assert or_condition.name == "(AlwaysTrue | AlwaysFalse)"


def test_condition_not_negates_operand() -> None:
    not_condition = ~always_true()
    assert not not_condition()
    assert not_condition.name == "~AlwaysTrue"


def test_condition_nested_composition() -> None:
    c1 = always_true()
    c2 = always_false()
    complex_condition = (c1 & c2) | ~c2
    assert complex_condition()
    assert complex_condition.name == "((AlwaysTrue & AlwaysFalse) | ~AlwaysFalse)"
