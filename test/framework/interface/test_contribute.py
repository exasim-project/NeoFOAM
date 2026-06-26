# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Unit tests for InterfaceSpec.contribute and _collect_contributions."""

from __future__ import annotations

import inspect

import pytest

from neofoam.framework.context import Context
from neofoam.framework.interface import Interface, InterfaceSpec
from framework.interface._helpers import VGREAT, _make_min_spec, _make_sum_spec


# ---------------------------------------------------------------------------
# @contribute registration
# ---------------------------------------------------------------------------


def test_contribute_decorator_returns_function_unchanged() -> None:
    spec = _make_min_spec("ret_unchanged")

    def my_contrib(delta_t: float) -> float:
        return delta_t * 0.5

    returned = spec.contribute(my_contrib)
    assert returned is my_contrib


def test_contribute_registers_function_on_spec() -> None:
    spec = _make_min_spec("reg_check")

    @spec.contribute
    def courant(delta_t: float) -> float:
        return delta_t * 0.5

    assert courant in spec._contributions


def test_multiple_contributions_all_registered() -> None:
    spec = _make_min_spec("multi_reg")

    @spec.contribute
    def first(a: float) -> float:
        return a

    @spec.contribute
    def second(b: float) -> float:
        return b

    assert first in spec._contributions
    assert second in spec._contributions
    assert len(spec._contributions) == 2


def test_contribute_can_be_added_to_spec_with_no_combine_registered() -> None:
    """Registration of a contribution is independent of whether @combine exists."""
    spec: InterfaceSpec[float] = Interface("no_combine_yet")

    @spec.contribute
    def contrib(x: float) -> float:
        return x

    assert contrib in spec._contributions


# ---------------------------------------------------------------------------
# _collect_contributions — happy paths
# ---------------------------------------------------------------------------


def test_collect_contributions_single_zero_arg_contribution() -> None:
    spec = _make_min_spec("single_zero_arg")

    @spec.contribute
    def always_five() -> float:
        return 5.0

    result = spec._collect_contributions({})
    assert result == pytest.approx(5.0)


def test_collect_contributions_single_contribution_with_param() -> None:
    spec = _make_min_spec("single_param")

    @spec.contribute
    def scaled(delta_t: float) -> float:
        return delta_t * 2.0

    result = spec._collect_contributions({"delta_t": 3.0})
    assert result == pytest.approx(6.0)


def test_collect_contributions_returns_fold_over_two_contributions() -> None:
    spec = _make_min_spec("two_contribs")

    @spec.contribute
    def courant(cfl_max: float) -> float:
        return cfl_max * 0.5

    @spec.contribute
    def diffusion(diff_limit: float) -> float:
        return diff_limit * 0.25

    result = spec._collect_contributions({"cfl_max": 4.0, "diff_limit": 8.0})
    # courant -> 2.0, diffusion -> 2.0 → min is 2.0
    assert result == pytest.approx(2.0)


def test_collect_contributions_min_selects_smallest() -> None:
    spec = _make_min_spec("min_select")

    @spec.contribute
    def fast_constraint(speed: float) -> float:
        return speed

    @spec.contribute
    def slow_constraint(drag: float) -> float:
        return drag

    result = spec._collect_contributions({"speed": 10.0, "drag": 3.0})
    assert result == pytest.approx(3.0)


def test_collect_contributions_no_contributions_returns_fold_default() -> None:
    """With zero contributions the fold is called with an empty iterable."""
    spec = _make_min_spec("zero_contrib")
    result = spec._collect_contributions({})
    assert result == VGREAT


def test_collect_contributions_sum_fold_two_contributions() -> None:
    spec = _make_sum_spec("sum_two")

    @spec.contribute
    def term_a(n: int) -> int:
        return n * 2

    @spec.contribute
    def term_b(m: int) -> int:
        return m * 3

    result = spec._collect_contributions({"n": 4, "m": 5})
    assert result == 8 + 15


def test_collect_contributions_each_call_invokes_functions_fresh() -> None:
    """_collect_contributions is side-effect free per call."""
    spec = _make_min_spec("fresh_call")
    counter: list[int] = [0]

    @spec.contribute
    def counting() -> float:
        counter[0] += 1
        return 1.0

    spec._collect_contributions({})
    spec._collect_contributions({})
    assert counter[0] == 2


def test_collect_contributions_contributions_share_providers_dict() -> None:
    """Multiple contributions may read from the same provider entry."""
    spec = _make_min_spec("shared_provider")

    @spec.contribute
    def cfl(dt: float) -> float:
        return dt * 0.1

    @spec.contribute
    def visc(dt: float) -> float:
        return dt * 0.2

    result = spec._collect_contributions({"dt": 10.0})
    # cfl -> 1.0, visc -> 2.0 → min is 1.0
    assert result == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _collect_contributions — error paths
# ---------------------------------------------------------------------------


def test_missing_provider_raises_value_error() -> None:
    spec = _make_min_spec("missing_prov")

    @spec.contribute
    def courant(phi: float, delta_t: float) -> float:
        return phi * delta_t

    with pytest.raises(ValueError):
        spec._collect_contributions({"phi": 1.0})  # delta_t missing


def test_missing_provider_error_names_the_missing_param() -> None:
    spec = _make_min_spec("missing_named")

    @spec.contribute
    def contrib(ghost_param: float) -> float:
        return ghost_param

    with pytest.raises(ValueError, match="ghost_param"):
        spec._collect_contributions({})


def test_missing_provider_error_names_the_spec() -> None:
    spec = _make_min_spec("named_spec_err")

    @spec.contribute
    def contrib(x: float) -> float:
        return x

    with pytest.raises(ValueError, match="named_spec_err"):
        spec._collect_contributions({})


def test_missing_provider_error_names_the_contribution() -> None:
    spec = _make_min_spec("named_contrib_err")

    @spec.contribute
    def my_special_contribution(x: float) -> float:
        return x

    with pytest.raises(ValueError, match="my_special_contribution"):
        spec._collect_contributions({})


def test_collect_contributions_without_combine_raises_runtime_error() -> None:
    spec: InterfaceSpec[float] = Interface("no_combine")

    @spec.contribute
    def contrib() -> float:
        return 1.0

    with pytest.raises(RuntimeError, match="combine"):
        spec._collect_contributions({})


# ---------------------------------------------------------------------------
# Parameter inference — no depends_on
# ---------------------------------------------------------------------------


def test_contribution_has_no_depends_on_attribute() -> None:
    """Contribution functions are plain callables — no depends_on machinery."""
    spec = _make_min_spec("plain_fn")

    @spec.contribute
    def contrib(dt: float) -> float:
        return dt

    # contrib is the raw function; it has no depends_on attribute
    assert not hasattr(contrib, "depends_on")


# ---------------------------------------------------------------------------
# Context-typed param rejected at registration time
# ---------------------------------------------------------------------------


def test_contribute_rejects_ctx_typed_param_at_registration() -> None:
    spec = _make_min_spec("no_ctx_param")

    with pytest.raises(ValueError):

        @spec.contribute
        def bad(ctx: Context) -> float:
            return 1.0


def test_contribute_rejects_ctx_param_message_names_param_and_spec() -> None:
    spec = _make_min_spec("ctx_msg_spec")

    with pytest.raises(ValueError, match="ctx") as exc_info:

        @spec.contribute
        def bad(ctx: Context) -> float:
            return 1.0

    assert "ctx_msg_spec" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Depends(callable) marker resolved by DependencyResolver
# ---------------------------------------------------------------------------


def test_contribution_resolves_depends_callable_marker() -> None:
    """A Depends(callable) marker in a contribution param is resolved by
    DependencyResolver, not the former name-only lookup."""
    from typing import Annotated

    from neofoam.framework.initialization.depends import Depends

    def provide_limit() -> float:
        return 0.05

    spec = _make_min_spec("depends_callable")

    # Define without the Annotated annotation first, then assign the live type
    # explicitly.  The module-level `from __future__ import annotations` would
    # otherwise store the annotation as a string, which the resolver cannot
    # inspect for Depends markers (the same constraint already applies to
    # @operation-defining modules in this codebase).
    def cfl_limit(dt: float) -> float:
        return dt

    cfl_limit.__annotations__ = {
        "dt": Annotated[float, Depends(provide_limit, cache=False)],
        "return": float,
    }

    spec.contribute(cfl_limit)

    # No provider for "dt" in the dict; DependencyResolver calls provide_limit()
    result = spec._collect_contributions({})
    assert result == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Extra/unused providers are silently ignored
# ---------------------------------------------------------------------------


def test_collect_contributions_ignores_extra_providers() -> None:
    spec = _make_min_spec("extra_providers")

    @spec.contribute
    def simple(a: float) -> float:
        return a

    result = spec._collect_contributions({"a": 1.0, "unused": 99.0})
    assert result == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Param-order independence (name-keyed, not positional)
# ---------------------------------------------------------------------------


def test_collect_contributions_param_order_independent() -> None:
    """Name-keyed binding: param declaration order must not affect the resolved value."""
    spec = _make_min_spec("param_order")

    @spec.contribute
    def asymmetric(b: float, a: float) -> float:
        return b - a  # non-commutative so a positional bug would surface

    result = spec._collect_contributions({"a": 3.0, "b": 10.0})
    assert result == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# Same-name collision — registry is a list, not name-keyed
# ---------------------------------------------------------------------------


def test_two_same_named_contributions_both_fold() -> None:
    """The registry is a list (identity-keyed), not a dict (name-keyed); both fold."""
    spec = _make_min_spec("same_name")

    def courant() -> float:
        return 2.0

    first = courant

    def courant() -> float:  # type: ignore[no-redef]
        return 5.0

    second = courant

    spec.contribute(first)
    spec.contribute(second)
    assert len(spec._contributions) == 2

    result = spec._collect_contributions({})
    assert result == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Defaulted params still require an explicit provider
# ---------------------------------------------------------------------------


def test_defaulted_param_still_requires_provider() -> None:
    """Python default values are NOT used as fallback — missing provider raises."""
    spec = _make_min_spec("default_no_fallback")

    @spec.contribute
    def with_default(x: float = 5.0) -> float:
        return x

    with pytest.raises(ValueError):
        spec._collect_contributions({})


# ---------------------------------------------------------------------------
# OCP — adding a contribution requires no edit to spec or consumer
# ---------------------------------------------------------------------------


def test_ocp_adding_second_contribution_needs_no_spec_change() -> None:
    """Adding a new @contribute changes nothing in the spec definition."""
    spec = _make_min_spec("ocp_test")

    # First contribution registered independently
    @spec.contribute
    def constraint_one(a: float) -> float:
        return a

    # "Consumer" snapshot: records which spec object it holds
    consumer_spec_ref = spec

    # Second contribution added later — spec object is unchanged
    @spec.contribute
    def constraint_two(b: float) -> float:
        return b

    # Consumer still holds the same spec, picks up the new contribution automatically
    assert consumer_spec_ref is spec
    result = consumer_spec_ref._collect_contributions({"a": 5.0, "b": 2.0})
    assert result == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Two-domain independence with @contribute
# ---------------------------------------------------------------------------


def test_two_domains_with_contribute_and_fold() -> None:
    """One mechanism, two different T types — no code change to the mechanism."""
    float_spec = _make_min_spec("float_domain")
    int_spec = _make_sum_spec("int_domain")

    @float_spec.contribute
    def cfl(dt: float) -> float:
        return dt * 0.1

    @float_spec.contribute
    def viscous(nu: float) -> float:
        return nu * 0.5

    @int_spec.contribute
    def source_a(n: int) -> int:
        return n * 10

    @int_spec.contribute
    def source_b(m: int) -> int:
        return m * 3

    float_result = float_spec._collect_contributions({"dt": 4.0, "nu": 2.0})
    int_result = int_spec._collect_contributions({"n": 3, "m": 4})

    assert float_result == pytest.approx(0.4)  # min(0.4, 1.0)
    assert int_result == 30 + 12  # sum(30, 12)


# ---------------------------------------------------------------------------
# self-skip: method-style contribution
# ---------------------------------------------------------------------------


def test_collect_contributions_skips_self_for_bound_method() -> None:
    """Bound-method contributions have 'self' already bound; the resolver skips it."""
    spec = _make_min_spec("method_style")

    class Limiter:
        def __init__(self, factor: float) -> None:
            self.factor = factor

        def constrain(self, dt: float) -> float:
            return dt * self.factor

    limiter = Limiter(0.5)
    spec.contribute(limiter.constrain)

    result = spec._collect_contributions({"dt": 4.0})
    assert result == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# self-skip on unbound functions (covers spec.py guard in contribute/collect)
# ---------------------------------------------------------------------------


def test_contribute_skips_self_on_unbound_function() -> None:
    """An unbound function with a 'self' first param registers and folds correctly.

    The 'self'/'cls' guard in contribute() is exercised.  The subtraction of
    'self' from fn_params in _collect_contributions is also exercised here.
    """
    spec = _make_min_spec("unbound_self")

    def constrain(self: object, dt: float) -> float:
        return dt * 0.5  # `self` ignored; result depends only on dt

    # contribute() must accept the unbound function (self/cls is skipped)
    spec.contribute(constrain)

    # _collect_contributions must not demand a "self" provider
    result = spec._collect_contributions({"dt": 4.0})
    assert result == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# get_type_hints fallback for unresolvable annotations
# ---------------------------------------------------------------------------


def test_contribute_tolerates_unresolvable_annotation() -> None:
    """A forward-ref annotation that get_type_hints cannot resolve does not raise.

    The except-branch fallback (hints={}) in contribute() is exercised.
    """
    spec = _make_min_spec("bad_hint")

    def contrib(x: float) -> float:
        return x

    # Plant an unresolvable forward reference on the annotation
    contrib.__annotations__["x"] = "NotDefinedAnywhere_XYZ"

    # contribute() must not raise — it falls back to hints={} and skips the Context check
    spec.contribute(contrib)
    assert contrib in spec._contributions


# ---------------------------------------------------------------------------
# Missing provider on a non-first contribution
# ---------------------------------------------------------------------------


def test_missing_provider_on_later_contribution_raises() -> None:
    """A missing provider on the second contribution still raises ValueError."""
    spec = _make_min_spec("later_missing")

    @spec.contribute
    def ok(a: float) -> float:
        return a

    @spec.contribute
    def bad(missing_param: float) -> float:
        return missing_param

    with pytest.raises(ValueError, match="bad"):
        spec._collect_contributions({"a": 1.0})


# ---------------------------------------------------------------------------
# inspect usage (kept as a lightweight smoke test)
# ---------------------------------------------------------------------------


def test_contribution_params_discoverable_via_inspect() -> None:
    """The resolver discovers params via inspect — same path as @operation."""
    spec = _make_min_spec("inspect_path")

    @spec.contribute
    def two_params(alpha: float, beta: float) -> float:
        return alpha + beta

    sig = inspect.signature(two_params)
    assert "alpha" in sig.parameters
    assert "beta" in sig.parameters
