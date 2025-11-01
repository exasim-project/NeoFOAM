import pytest
from foamadapter.framework.dag import StepNumber

def test_step_number():
    # Basic comparisons
    assert StepNumber("1") < StepNumber("1.10")
    assert StepNumber("1") == StepNumber(1)
    assert StepNumber("1.10.0") > StepNumber("1.9.0")
    assert StepNumber("1.10.0") < StepNumber("1.10.1")
    assert StepNumber("1") > StepNumber("1.-1")

    # Equality and padding
    assert StepNumber("1.10") == StepNumber("1.10.0")
    assert StepNumber("1.0.0") == StepNumber("1")
    assert StepNumber("1.0.1") != StepNumber("1")

    # Mixed input types
    assert StepNumber([1, 2, 3]) == StepNumber("1.2.3")
    assert StepNumber((1, 2)) < StepNumber("1.2.1")

    # Greater / less / equal checks
    a = StepNumber("2.0")
    b = StepNumber("1.9.9")
    c = StepNumber("2.0.0")
    assert a > b
    assert a >= c
    assert c <= a
    assert not a < b

    # Invalid inputs
    with pytest.raises(TypeError):
        StepNumber(1.23)
    with pytest.raises(ValueError):
        StepNumber("1.a.3")