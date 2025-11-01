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

    assert c1() == True
    assert c2() == False

    and_condition = c1 & c2
    assert and_condition() == False
    assert and_condition._name == "(AlwaysTrue & AlwaysFalse)"

    or_condition = c1 | c2
    assert or_condition() == True
    assert or_condition._name == "(AlwaysTrue | AlwaysFalse)"

    not_condition = ~c1
    assert not_condition() == False
    assert not_condition._name == "~AlwaysTrue"
    complex_condition = (c1 & c2) | ~c2
    assert complex_condition() == True
    assert complex_condition._name == "((AlwaysTrue & AlwaysFalse) | ~AlwaysFalse)"


class MaxIterationsCondition(Condition):
    def __init__(self, max_iterations: int):
        super().__init__(self.check, f"MaxIterations({max_iterations})")
        self.max_iterations = max_iterations
        self.current_iteration = 0

    def check(self) -> bool:
        self.current_iteration += 1
        return self.current_iteration <= self.max_iterations    
    

def test_max_iterations_condition():
    max_iter_cond = MaxIterationsCondition(3)

    assert max_iter_cond() == True  # 1
    assert max_iter_cond() == True  # 2
    assert max_iter_cond() == True  # 3
    assert max_iter_cond() == False # 4
    assert max_iter_cond() == False # 5