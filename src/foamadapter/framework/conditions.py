# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors
"""
Unified Condition class with logical operators.
"""

from __future__ import annotations

from typing import Callable


class Condition:
    """
    Unified condition class with logical operators.

    Usage:
        c1 = Condition( lambda: True, "AlwaysTrue")
        c2 = Condition(lambda: check_something(), "MyCheck")
        combined = (c1 & c2) | ~c1
    """

    def __init__(self, condition_func: Callable[..., bool], name: str = "Condition"):
        self._condition_func = condition_func
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def __call__(self, *args: object, **kwargs: object) -> bool:
        return self._condition_func(*args, **kwargs)

    def __and__(self, other: Condition) -> Condition:
        def combined_condition(*args: object, **kwargs: object) -> bool:
            return self(*args, **kwargs) and other(*args, **kwargs)

        return Condition(combined_condition, f"({self._name} & {other._name})")

    def __or__(self, other: Condition) -> Condition:
        def combined_condition(*args: object, **kwargs: object) -> bool:
            return self(*args, **kwargs) or other(*args, **kwargs)

        return Condition(combined_condition, f"({self._name} | {other._name})")

    def __invert__(self) -> Condition:
        def inverted_condition(*args: object, **kwargs: object) -> bool:
            return not self(*args, **kwargs)

        return Condition(inverted_condition, f"~{self._name}")
