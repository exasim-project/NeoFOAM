# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Unit tests for InterfaceSpec and the Interface factory."""

from __future__ import annotations

import builtins
import sys
from typing import cast

import pytest

from neofoam.framework.interface import Interface, InterfaceSpec
from framework.interface._helpers import VGREAT, _make_min_spec


# ---------------------------------------------------------------------------
# Public import smoke test
# ---------------------------------------------------------------------------


def test_top_level_import_exposes_interface_and_spec() -> None:
    from neofoam.framework import Interface as TopInterface
    from neofoam.framework import InterfaceSpec as TopInterfaceSpec

    x = TopInterface("x")
    assert x.name == "x"
    assert isinstance(x, TopInterfaceSpec)


# ---------------------------------------------------------------------------
# Factory and identity
# ---------------------------------------------------------------------------


def test_interface_factory_returns_spec() -> None:
    spec = Interface("timeStepConstraint")
    assert isinstance(spec, InterfaceSpec)
    assert spec.name == "timeStepConstraint"


def test_interface_factory_assigns_name_verbatim() -> None:
    spec = Interface("momentumSource")
    assert spec.name == "momentumSource"


def test_interface_factory_rejects_extra_kwargs() -> None:
    with pytest.raises(TypeError):
        Interface("x", combine=min)  # type: ignore[call-arg]


def test_two_factories_produce_independent_specs() -> None:
    a = Interface("alpha")
    b = Interface("beta")
    assert a is not b
    assert a.name != b.name


# ---------------------------------------------------------------------------
# @combine registration
# ---------------------------------------------------------------------------


def test_combine_decorator_returns_original_function() -> None:
    spec = Interface("x")

    def my_combine(values: object) -> float:
        return 0.0

    returned = spec.combine(my_combine)
    assert returned is my_combine


def test_second_combine_raises_at_registration() -> None:
    spec = Interface("y")

    @spec.combine
    def first(values: object) -> float:
        return 0.0

    with pytest.raises(RuntimeError, match="combine"):

        @spec.combine
        def second(values: object) -> float:
            return 1.0


def test_no_combine_registered_message_mentions_combine() -> None:
    """RuntimeError from duplicate @combine must mention 'combine' in the message."""
    spec = Interface("dup_check")

    @spec.combine
    def first(values: object) -> float:
        return 0.0

    with pytest.raises(RuntimeError) as exc_info:

        @spec.combine
        def second(values: object) -> float:
            return 1.0

    assert "combine" in str(exc_info.value).lower()


def test_duplicate_combine_error_contains_spec_name() -> None:
    spec = Interface("dupe_name_check")

    @spec.combine
    def first(values: object) -> float:
        return 0.0

    with pytest.raises(RuntimeError) as exc_info:

        @spec.combine
        def second(values: object) -> float:
            return 1.0

    assert "dupe_name_check" in str(exc_info.value)


# ---------------------------------------------------------------------------
# _collect_values fold behaviour
# ---------------------------------------------------------------------------


def test_fold_over_empty_providers_returns_default() -> None:
    spec = _make_min_spec("tsc")
    result = spec._collect_values([])
    assert result == VGREAT


def test_fold_over_single_provider() -> None:
    spec = _make_min_spec("single")
    result = spec._collect_values([lambda: 0.5])
    assert result == pytest.approx(0.5)


def test_fold_over_two_providers_returns_min() -> None:
    spec = _make_min_spec("two_providers")
    result = spec._collect_values([lambda: 3.0, lambda: 1.5])
    assert result == pytest.approx(1.5)


def test_fold_over_many_providers_returns_global_min() -> None:
    spec = _make_min_spec("many")
    providers = [lambda v=v: float(v) for v in [10.0, 3.0, 7.0, 1.0, 5.0]]
    result = spec._collect_values(providers)
    assert result == pytest.approx(1.0)


def test_fold_is_deterministic_on_reinvocation() -> None:
    spec = _make_min_spec("reinvoke")
    providers = [lambda: 2.0, lambda: 4.0]
    first_result = spec._collect_values(providers)
    second_result = spec._collect_values(providers)
    assert first_result == second_result


def test_provider_callable_is_called_each_time() -> None:
    """Each _collect_values call invokes all providers fresh."""
    spec: InterfaceSpec[float] = Interface("freshcall")

    @spec.combine
    def fold(values: object) -> float:
        total = 0.0
        for v in values:  # type: ignore[union-attr]
            total += v
        return total

    counter = [0]

    def counting_provider() -> float:
        counter[0] += 1
        return 1.0

    spec._collect_values([counting_provider])
    spec._collect_values([counting_provider])
    assert counter[0] == 2


def test_fold_without_combine_registered_raises() -> None:
    spec: InterfaceSpec[float] = Interface("nocombine")
    with pytest.raises(RuntimeError, match=r"(?i)combine"):
        spec._collect_values([lambda: 1.0])


def test_fold_without_combine_error_contains_spec_name() -> None:
    spec: InterfaceSpec[float] = Interface("my_named_spec")
    with pytest.raises(RuntimeError) as exc_info:
        spec._collect_values([lambda: 1.0])
    assert "my_named_spec" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------


def test_fold_ordering_is_stable_for_list_append_fold() -> None:
    """A fold that accumulates results preserves provider order."""
    spec: InterfaceSpec[list[str]] = Interface("ordered")

    @spec.combine
    def fold(values: object) -> list[str]:
        result: list[str] = []
        for v in values:  # type: ignore[union-attr]
            result.extend(v)
        return result

    providers = [lambda: ["a"], lambda: ["b"], lambda: ["c"]]
    result = spec._collect_values(providers)
    assert result == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Two-domain independence (same mechanism, different T)
# ---------------------------------------------------------------------------


def test_sum_fold_domain_works_with_same_mechanism() -> None:
    spec: InterfaceSpec[int] = Interface("intSum")

    @spec.combine
    def fold(values: object) -> int:
        return sum(values)  # type: ignore[call-overload]

    result = spec._collect_values([lambda: 3, lambda: 7])
    assert result == 10


def test_two_independent_specs_do_not_share_combine_state() -> None:
    min_spec: InterfaceSpec[float] = Interface("minDomain")
    sum_spec: InterfaceSpec[float] = Interface("sumDomain")

    @min_spec.combine
    def fold_min(values: object) -> float:
        return cast(float, builtins.min(values, default=0.0))  # type: ignore[call-overload]

    @sum_spec.combine
    def fold_sum(values: object) -> float:
        return sum(values)  # type: ignore[call-overload]

    min_result = min_spec._collect_values([lambda: 3.0, lambda: 1.0])
    sum_result = sum_spec._collect_values([lambda: 3.0, lambda: 1.0])
    assert min_result == pytest.approx(1.0)
    assert sum_result == pytest.approx(4.0)


def test_registering_combine_on_one_spec_does_not_affect_other() -> None:
    a = Interface("specA")
    b = Interface("specB")

    @a.combine
    def fold_a(values: object) -> float:
        return 0.0

    # b still has no combine — calling _collect_values should raise
    with pytest.raises(RuntimeError):
        b._collect_values([])


# ---------------------------------------------------------------------------
# Pure-Python / no backend import
# ---------------------------------------------------------------------------


def test_interface_module_is_pure_python() -> None:
    pybfoam_before = "pybFoam" in sys.modules
    import neofoam.framework.interface  # noqa: F401

    pybfoam_after = "pybFoam" in sys.modules
    assert pybfoam_before == pybfoam_after, (
        "importing neofoam.framework.interface must not pull in pybFoam"
    )
