# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Owner-gated contribution folding.

A contribution folds via ``collect(ctx)`` iff its owning model's name is present
in ``ctx.models``; an unowned contribution always folds. There is no mutable
active-set — gating is a pure function of (registered contributions, ctx).
"""

# NOTE: no `from __future__ import annotations` — contribution params are resolved
# by name from live annotations at fold time (mirrors the sibling interface tests).

import pytest

from neofoam.framework.context import Context
from neofoam.framework.interface import InterfaceSpec
from neofoam.framework.model import Model
from framework.interface._helpers import VGREAT, _make_min_spec


def test_unowned_contribution_folds_regardless_of_active_models() -> None:
    spec = _make_min_spec("unowned")

    @spec.contribute
    def cap(dt: float) -> float:
        return dt

    ctx = Context(fields={"dt": 5.0}, models={})
    assert spec.collect(ctx) == pytest.approx(5.0)


def test_owned_contribution_folds_when_model_active() -> None:
    spec = _make_min_spec("owned_active")
    courant = Model("courant")

    @spec.contribute(model=courant)
    def cap(dt: float) -> float:
        return dt

    ctx = Context(fields={"dt": 3.0}, models={"courant": object()})
    assert spec.collect(ctx) == pytest.approx(3.0)


def test_owned_contribution_registered_but_excluded_when_model_inactive() -> None:
    spec = _make_min_spec("owned_inactive")
    courant = Model("courant")

    @spec.contribute(model=courant)
    def cap(dt: float) -> float:
        return dt

    # Registered (discoverable) but its owner is not active for this case.
    assert cap in spec._contributions
    ctx = Context(fields={"dt": 3.0}, models={})
    assert spec.collect(ctx) == VGREAT


def test_two_owned_contributions_gate_independently() -> None:
    spec = _make_min_spec("two_owners")
    courant = Model("courant")
    max_delta_t = Model("maxDeltaT")

    @spec.contribute(model=courant)
    def courant_cap(dt: float) -> float:
        return dt

    @spec.contribute(model=max_delta_t)
    def fixed_cap() -> float:
        # Smaller than the active model's value: if it wrongly folded it would win
        # the min, so a gating regression changes the result.
        return 0.01

    ctx = Context(fields={"dt": 0.2}, models={"courant": object()})
    assert spec.collect(ctx) == pytest.approx(0.2)


def test_same_spec_two_contexts_do_not_leak() -> None:
    spec = _make_min_spec("no_leak")
    courant = Model("courant")
    max_delta_t = Model("maxDeltaT")

    @spec.contribute(model=courant)
    def courant_cap(dt: float) -> float:
        return dt

    @spec.contribute(model=max_delta_t)
    def fixed_cap() -> float:
        return 0.01

    # ctx_a: courant active (0.2); maxDeltaT excluded but would give 0.01 < 0.2,
    # so a leak there changes the fold.
    ctx_a = Context(fields={"dt": 0.2}, models={"courant": object()})
    # ctx_b: maxDeltaT active (0.01); courant excluded but would give 0.005 < 0.01,
    # so a leak there changes the fold.
    ctx_b = Context(fields={"dt": 0.005}, models={"maxDeltaT": object()})

    assert spec.collect(ctx_a) == pytest.approx(0.2)
    assert spec.collect(ctx_b) == pytest.approx(0.01)
    assert spec.collect(ctx_a) == pytest.approx(0.2)


def test_no_active_model_folds_to_default() -> None:
    spec = _make_min_spec("fixed_step")
    courant = Model("courant")

    @spec.contribute(model=courant)
    def cap(dt: float) -> float:
        return dt

    # No time-step model configured for this case -> fixed step (VGREAT).
    ctx = Context(fields={"dt": 1.0}, models={})
    assert spec.collect(ctx) == VGREAT


def test_contribute_with_model_returns_function_unchanged() -> None:
    spec = _make_min_spec("ret_unchanged")
    courant = Model("courant")

    def my_contrib(dt: float) -> float:
        return dt

    returned = spec.contribute(my_contrib, model=courant)
    assert returned is my_contrib


def test_activate_deactivate_and_active_set_are_removed() -> None:
    spec: InterfaceSpec[float] = _make_min_spec("removed")
    assert not hasattr(spec, "activate")
    assert not hasattr(spec, "deactivate")
    assert not hasattr(spec, "_active_contributions")


def test_contribute_rejects_a_model_without_name() -> None:
    spec = _make_min_spec("bad_owner")
    with pytest.raises(TypeError, match="no '.name'"):

        @spec.contribute(model=object())
        def cap(dt: float) -> float:
            return dt


def test_owned_bound_method_contribution_folds_through_collect() -> None:
    spec = _make_min_spec("bound_self")
    courant = Model("courant")

    class Holder:
        @spec.contribute(model=courant)
        def cap(self, dt: float) -> float:
            return dt

    ctx = Context(fields={"dt": 4.0}, models={"courant": object()})
    assert spec.collect(ctx) == pytest.approx(4.0)


def test_two_contributions_sharing_one_owner_both_fold_when_active() -> None:
    spec = _make_min_spec("shared_owner")
    courant = Model("courant")

    @spec.contribute(model=courant)
    def cap_a(dt: float) -> float:
        return dt * 0.5

    @spec.contribute(model=courant)
    def cap_b(dt: float) -> float:
        return dt * 0.25

    ctx_on = Context(fields={"dt": 4.0}, models={"courant": object()})
    assert spec.collect(ctx_on) == pytest.approx(1.0)  # min(2.0, 1.0)

    ctx_off = Context(fields={"dt": 4.0}, models={})
    assert spec.collect(ctx_off) == VGREAT
