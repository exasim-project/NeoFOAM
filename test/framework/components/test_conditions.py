"""
Tests for the unified condition system.
"""

from foamadapter.framework.conditions import Condition


def always_true():
    return Condition(lambda: True, "AlwaysTrue")


def always_false():
    return Condition(lambda: False, "AlwaysFalse")


def test_conditions():
    c1 = always_true()
    c2 = always_false()

    assert c1()
    assert not c2()

    and_condition = c1 & c2
    assert not and_condition()
    assert and_condition.name == "(AlwaysTrue & AlwaysFalse)"

    or_condition = c1 | c2
    assert or_condition()
    assert or_condition.name == "(AlwaysTrue | AlwaysFalse)"

    not_condition = ~c1
    assert not not_condition()
    assert not_condition.name == "~AlwaysTrue"
    complex_condition = (c1 & c2) | ~c2
    assert complex_condition()
    assert complex_condition.name == "((AlwaysTrue & AlwaysFalse) | ~AlwaysFalse)"
