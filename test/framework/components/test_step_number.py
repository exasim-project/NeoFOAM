import pytest

from foamadapter.framework.types import OperationNumber


def test_step_number():
    # Basic comparisons
    assert OperationNumber("1") < OperationNumber("1.10")
    assert OperationNumber("1") == OperationNumber(1)
    assert OperationNumber("1.10.0") > OperationNumber("1.9.0")
    assert OperationNumber("1.10.0") < OperationNumber("1.10.1")
    assert OperationNumber("1") > OperationNumber("1.-1")

    # Equality and padding
    assert OperationNumber("1.10") == OperationNumber("1.10.0")
    assert OperationNumber("1.0.0") == OperationNumber("1")
    assert OperationNumber("1.0.1") != OperationNumber("1")

    # Mixed input types
    assert OperationNumber([1, 2, 3]) == OperationNumber("1.2.3")
    assert OperationNumber((1, 2)) < OperationNumber("1.2.1")

    # Greater / less / equal checks
    a = OperationNumber("2.0")
    b = OperationNumber("1.9.9")
    c = OperationNumber("2.0.0")
    assert a > b
    assert a >= c
    assert c <= a
    assert not a < b

    # Invalid inputs
    with pytest.raises(TypeError):
        OperationNumber(1.23)
    with pytest.raises(ValueError):
        OperationNumber("1.a.3")
