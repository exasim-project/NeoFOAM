"""
Unified Condition class with logical operators.
"""

from typing import Callable


class Condition:
    """
    Unified condition class with logical operators.

    Usage:
        c1 = Condition.max_iterations(100)
        c2 = Condition(lambda: check_something(), "MyCheck")
        combined = (c1 & c2) | ~c1
    """

    def __init__(self, condition_func: Callable[..., bool], name: str = "Condition"):
        self._condition_func = condition_func
        self._name = name

    def __call__(self, *args, **kwargs) -> bool:
        return self._condition_func(*args, **kwargs)

    def __and__(self, other: "Condition") -> "Condition":
        def combined_condition(*args, **kwargs):
            return self(*args, **kwargs) and other(*args, **kwargs)

        return Condition(combined_condition, f"({self._name} & {other._name})")

    def __or__(self, other: "Condition") -> "Condition":
        def combined_condition(*args, **kwargs):
            return self(*args, **kwargs) or other(*args, **kwargs)

        return Condition(combined_condition, f"({self._name} | {other._name})")

    def __invert__(self) -> "Condition":
        def inverted_condition(*args, **kwargs):
            return not self(*args, **kwargs)

        return Condition(inverted_condition, f"~{self._name}")
