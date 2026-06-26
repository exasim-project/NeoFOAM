# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for ctx.interfaces placement, active-contribution gating,
consumer injection by spec-typed param, and spec.collect(ctx)."""

# NOTE: Do NOT add `from __future__ import annotations` to this module.
# Consumer functions annotated with an InterfaceSpec instance as a param type
# must have live (non-string) annotations at inspection time.  The
# `from __future__ import annotations` PEP 563 mechanism stringifies all
# annotations, which breaks `isinstance(param.annotation, InterfaceSpec)`.

import pytest

from neofoam.framework.context import Context
from neofoam.framework.dependency_resolver import DependencyResolver
from neofoam.framework.initialization import execute_initialization, interface_step
from neofoam.framework.interface.spec import BoundInterface
from framework.interface._helpers import VGREAT, _make_min_spec, _make_sum_spec


# ---------------------------------------------------------------------------
# Task B — interface_step + ctx.interfaces routing
# ---------------------------------------------------------------------------


def test_interface_step_creates_init_step_with_interfaces_category() -> None:
    step = interface_step("tsc", create=lambda _ctx: "sentinel")
    assert step.name == "interfaces.tsc"
    assert step.category == "interfaces"


def test_interface_step_value_lands_in_ctx_interfaces() -> None:
    sentinel = object()
    step = interface_step("myIface", create=lambda _ctx: sentinel)
    ctx = execute_initialization([step])
    assert ctx.interfaces["myIface"] is sentinel


# ---------------------------------------------------------------------------
# Task C — _active_contributions and activate/deactivate
# ---------------------------------------------------------------------------


def test_all_contributions_active_by_default() -> None:
    spec = _make_min_spec("default_active")

    @spec.contribute
    def c1(x: float) -> float:
        return x

    assert c1 in spec._active_contributions


def test_deactivate_excludes_contribution_from_collect_contributions() -> None:
    spec = _make_min_spec("deactivate")

    @spec.contribute
    def small(x: float) -> float:
        return x

    @spec.contribute
    def large(y: float) -> float:
        return y

    spec.deactivate(small)
    result = spec._collect_contributions({"x": 1.0, "y": 100.0})
    # small is inactive — only large folds, so result is min([100.0]) = 100.0
    assert result == pytest.approx(100.0)


def test_activate_re_enables_deactivated_contribution() -> None:
    spec = _make_min_spec("reactivate")

    @spec.contribute
    def c(x: float) -> float:
        return x

    spec.deactivate(c)
    spec.activate(c)
    result = spec._collect_contributions({"x": 3.0})
    assert result == pytest.approx(3.0)


def test_activate_unknown_fn_raises_value_error() -> None:
    spec = _make_min_spec("activate_unknown")

    def not_registered(x: float) -> float:
        return x

    with pytest.raises(ValueError, match="activate"):
        spec.activate(not_registered)


# ---------------------------------------------------------------------------
# Task C — collect(ctx) resolves params from Context.fields
# ---------------------------------------------------------------------------


def test_collect_with_context_resolves_annotated_field_param() -> None:
    """collect(ctx) resolves a field-typed param from ctx.fields."""
    spec = _make_min_spec("ctx_fields")

    def cfl(dt: float) -> float:
        return dt * 0.5

    spec.contribute(cfl)
    ctx = Context(fields={"dt": 4.0}, models={})
    result = spec.collect(ctx)
    assert result == pytest.approx(2.0)


def test_collect_with_inactive_contribution_excluded() -> None:
    spec = _make_min_spec("ctx_inactive")

    def big(dt: float) -> float:
        return dt * 10.0

    def small(dt: float) -> float:
        return dt * 0.1

    spec.contribute(big)
    spec.contribute(small)
    spec.deactivate(big)

    ctx = Context(fields={"dt": 2.0}, models={})
    result = spec.collect(ctx)
    # only small is active: 2.0 * 0.1 = 0.2
    assert result == pytest.approx(0.2)


def test_spec_collect_with_no_active_contributions_returns_empty_fold() -> None:
    """collect(ctx) with zero active contributions returns the combine's empty-iterable result."""
    spec = _make_min_spec("empty_collect")
    ctx = Context(fields={}, models={})
    result = spec.collect(ctx)
    assert result == VGREAT  # min([], default=VGREAT)


# ---------------------------------------------------------------------------
# Task D — DependencyResolver injects BoundInterface for spec-annotated params
# ---------------------------------------------------------------------------


def test_resolver_injects_bound_interface_for_spec_annotated_param() -> None:
    """A param annotated with an InterfaceSpec instance is injected as BoundInterface."""
    spec = _make_min_spec("inject_iface")

    @spec.contribute
    def c(x: float) -> float:
        return x

    ctx = Context(fields={}, models={}, interfaces={"inject_iface": spec})

    def consumer(constraints: spec) -> float:  # type: ignore[valid-type]
        return constraints()

    resolver = DependencyResolver()
    kwargs = resolver.resolve_arguments(consumer, ctx=ctx, x=2.0)
    assert "constraints" in kwargs
    assert isinstance(kwargs["constraints"], BoundInterface)


def test_bound_interface_call_returns_fold() -> None:
    spec = _make_min_spec("bound_call")

    @spec.contribute
    def c(x: float) -> float:
        return x

    ctx = Context(fields={"x": 5.0}, models={}, interfaces={"bound_call": spec})

    def consumer(constraints: spec) -> float:  # type: ignore[valid-type]
        return constraints()

    resolver = DependencyResolver()
    kwargs = resolver.resolve_arguments(consumer, ctx=ctx)
    result = kwargs["constraints"]()
    assert result == pytest.approx(5.0)


def test_resolver_injects_two_interfaces_independently() -> None:
    """Two distinct InterfaceSpec params on one consumer function are injected separately."""
    spec_a = _make_min_spec("iface_a")
    spec_b = _make_sum_spec("iface_b")

    @spec_a.contribute
    def ca(x: float) -> float:
        return x

    @spec_b.contribute
    def cb(n: int) -> int:
        return n

    ctx = Context(
        fields={"x": 3.0, "n": 7},
        models={},
        interfaces={"iface_a": spec_a, "iface_b": spec_b},
    )

    def consumer(a: spec_a, b: spec_b) -> None:  # type: ignore[valid-type]
        pass

    resolver = DependencyResolver()
    kwargs = resolver.resolve_arguments(consumer, ctx=ctx)
    assert isinstance(kwargs["a"], BoundInterface)
    assert isinstance(kwargs["b"], BoundInterface)
    assert kwargs["a"]() == pytest.approx(3.0)
    assert kwargs["b"]() == 7


def test_resolver_raises_when_interface_not_in_ctx() -> None:
    spec = _make_min_spec("missing_iface")
    ctx = Context(fields={}, models={}, interfaces={})  # spec NOT placed

    def consumer(constraints: spec) -> float:  # type: ignore[valid-type]
        return constraints()

    resolver = DependencyResolver()
    with pytest.raises(ValueError, match="missing_iface"):
        resolver.resolve_arguments(consumer, ctx=ctx)


# ---------------------------------------------------------------------------
# Task E — End-to-end: interface_step + collect + inactive-contribution gating
# ---------------------------------------------------------------------------


def test_collect_via_init_graph_and_interface_step() -> None:
    """End-to-end: interface_step places spec in ctx.interfaces; collect returns fold."""
    spec = _make_min_spec("e2e_tsc")

    @spec.contribute
    def cfl(dt: float) -> float:
        return dt * 0.5

    # Init graph: place the spec in ctx.interfaces
    step = interface_step("e2e_tsc", create=lambda _ctx: spec)
    ctx = execute_initialization([step])

    # Verify spec landed in ctx.interfaces
    assert "e2e_tsc" in ctx.interfaces
    assert ctx.interfaces["e2e_tsc"] is spec

    # spec.collect(ctx) with fields populated
    ctx_with_fields = Context(
        fields={"dt": 4.0},
        models={},
        interfaces={"e2e_tsc": spec},
    )
    result = spec.collect(ctx_with_fields)
    assert result == pytest.approx(2.0)


def test_inactive_model_contribution_excluded_via_deactivate() -> None:
    """Deactivating a contribution before collect excludes it from the fold."""
    spec = _make_min_spec("if13_gate")

    @spec.contribute
    def active_contrib(dt: float) -> float:
        return dt

    @spec.contribute
    def inactive_contrib(dt: float) -> float:
        return dt * 0.001  # would be smallest if active

    spec.deactivate(inactive_contrib)
    ctx = Context(fields={"dt": 5.0}, models={})
    result = spec.collect(ctx)
    # inactive_contrib excluded → only active_contrib folds → 5.0
    assert result == pytest.approx(5.0)
