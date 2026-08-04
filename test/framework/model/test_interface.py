# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for model-owned interfaces: ``@<model>.interface`` declares a gather
:class:`Hook` on a model-private extension — the declaration body receives the
active contributions' results and combines them. Covers declaration, empty-fold
default, contribution registration, live-Context dispatch, and resolver
injection of a hook-annotated parameter."""

# NOTE: no `from __future__ import annotations` — a Hook handle is used as a live
# operation-parameter annotation; keep these modules PEP 563-free so annotations
# stay non-string objects.

from typing import Annotated, Any, Callable, Iterable

import pytest

from neofoam.framework.context import Context
from neofoam.framework.dependency_resolver import DependencyResolver
from neofoam.framework.initialization.depends import Depends
from neofoam.framework.model import Hook, Model
from neofoam.framework.model.runtime import ModelRuntime
from neofoam.io import BaseConfig

VGREAT = 1e300


class _CapConfig(BaseConfig):
    value: float


def _loop_with_constraint() -> Any:
    loop = Model("solutionLoop")

    @loop.interface
    def timeStepConstraint(limits: Iterable[float]) -> float:
        return min(limits, default=VGREAT)

    return timeStepConstraint


@pytest.fixture
def constraint() -> Any:
    """A model-owned ``timeStepConstraint`` interface (min-fold, VGREAT default)."""
    return _loop_with_constraint()


def contributor(
    interface: Any, fn: Callable[..., Any], name: str, config: Any = None
) -> ModelRuntime:
    """A fresh ``Model`` named *name* contributing *fn* to *interface*, returned
    as the live ``ModelRuntime`` that makes the contribution active for a case."""
    model = Model(name)
    model.contributes(interface)(fn)
    return ModelRuntime(spec=model, name=name, config=config)


def bound(interface: Any, runtimes: Iterable[ModelRuntime], fields: Any = None) -> Any:
    """*interface* resolved against a Context in which exactly *runtimes* are
    active — the injected form a consumer receives and calls."""
    ctx = Context(fields=dict(fields or {}), models={rt.name: rt for rt in runtimes})
    return interface.resolve(ctx)


# Named, annotated contributions reused by the config / error-message families:
# their `__name__` and `cfg: _CapConfig` annotation are load-bearing (unlike the
# anonymous lambdas used where only the folded value matters).
def _needs_deltaT(deltaT: float) -> float:
    return deltaT


def _cap_from_config(cfg: _CapConfig) -> float:
    return cfg.value


# ---------------------------------------------------------------------------
# Declaring an interface on a model
# ---------------------------------------------------------------------------


def test_interface_decorator_returns_a_hook_named_after_the_declaration(
    constraint: Any,
) -> None:
    # An interface IS an extension hook — one mechanism, two spellings.
    assert isinstance(constraint, Hook)
    assert constraint.name == "timeStepConstraint"


def test_handle_round_trips_through_its_owning_model() -> None:
    loop = Model("solutionLoop")

    @loop.interface
    def timeStepConstraint(limits: Iterable[float]) -> float:
        return min(limits, default=VGREAT)

    assert loop.declared_interfaces["timeStepConstraint"] is timeStepConstraint


def test_redeclaring_the_same_interface_name_is_rejected() -> None:
    loop = Model("solutionLoop")

    def declare() -> None:
        @loop.interface
        def timeStepConstraint(limits: Iterable[float]) -> float:
            return min(limits, default=VGREAT)

    declare()
    with pytest.raises(RuntimeError, match="already declared"):
        declare()


def test_same_interface_name_on_two_models_is_allowed() -> None:
    loop_a = Model("loopA")
    loop_b = Model("loopB")

    @loop_a.interface
    def timeStepConstraint(limits: Iterable[float]) -> float:
        return min(limits, default=VGREAT)

    @loop_b.interface
    def timeStepConstraint(limits: Iterable[float]) -> float:  # noqa: F811
        return min(limits, default=VGREAT)

    hook_a = loop_a.declared_interfaces["timeStepConstraint"]
    hook_b = loop_b.declared_interfaces["timeStepConstraint"]
    assert hook_a is not hook_b


@pytest.mark.parametrize(
    "limits, expected",
    [
        ([], VGREAT),  # empty -> the documented default
        ([2.0, 1.0, 3.0], 1.0),  # combines supplied values
        ([5.0], 5.0),  # single value
        ([1.0, 1.0], 1.0),  # duplicates
        ((x for x in (3.0, 2.0, 4.0)), 2.0),  # a generator
    ],
)
def test_the_declaration_body_reduces_inputs_via_the_declared_combine(
    constraint: Any, limits: Iterable[float], expected: float
) -> None:
    assert constraint.declaration(limits) == expected


# ---------------------------------------------------------------------------
# Registering contributions (recorded, not yet folded)
# ---------------------------------------------------------------------------


def test_contributes_registers_the_function_against_the_interface(
    constraint: Any,
) -> None:
    courant = Model("courant")

    @courant.contributes(constraint)
    def courant_limit(deltaT: float) -> float:
        return deltaT * 0.5

    assert courant_limit in constraint.contributions


def test_contributes_records_the_contributing_model_as_owner(constraint: Any) -> None:
    courant = Model("courant")

    @courant.contributes(constraint)
    def courant_limit(deltaT: float) -> float:
        return deltaT * 0.5

    assert constraint.owner_of(courant_limit) is courant


def test_contributes_returns_the_function_unchanged(constraint: Any) -> None:
    courant = Model("courant")

    def courant_limit(deltaT: float) -> float:
        return deltaT * 0.5

    returned = courant.contributes(constraint)(courant_limit)
    assert returned is courant_limit


def test_two_models_contribute_independently_to_one_interface(constraint: Any) -> None:
    courant = Model("courant")
    max_delta_t = Model("maxDeltaT")

    @courant.contributes(constraint)
    def courant_limit(deltaT: float) -> float:
        return deltaT

    @max_delta_t.contributes(constraint)
    def fixed_cap() -> float:
        return 0.01

    assert set(constraint.contributions) == {courant_limit, fixed_cap}
    assert constraint.owner_of(courant_limit) is courant
    assert constraint.owner_of(fixed_cap) is max_delta_t


def test_contributions_are_returned_in_registration_order(constraint: Any) -> None:
    courant = Model("courant")
    max_delta_t = Model("maxDeltaT")

    @courant.contributes(constraint)
    def courant_limit(deltaT: float) -> float:
        return deltaT

    @max_delta_t.contributes(constraint)
    def fixed_cap() -> float:
        return 0.01

    assert constraint.contributions == (courant_limit, fixed_cap)


def test_registration_does_not_yet_participate_in_the_fold(constraint: Any) -> None:
    # Registration is recorded for later gathering but is inert here: with no
    # active runtime the fold still returns the default even after a
    # contribution is registered.
    courant = Model("courant")

    @courant.contributes(constraint)
    def courant_limit(deltaT: float) -> float:
        return 0.001

    assert bound(constraint, [])() == VGREAT


def test_contributes_against_a_non_hook_target_raises() -> None:
    courant = Model("courant")
    with pytest.raises(TypeError, match="must be a hook"):

        @courant.contributes(lambda limits: min(limits))  # type: ignore[arg-type]
        def bad(deltaT: float) -> float:
            return deltaT


def test_owner_of_unregistered_function_raises(constraint: Any) -> None:
    def never_registered(deltaT: float) -> float:
        return deltaT

    with pytest.raises(KeyError, match="not a registered contribution"):
        constraint.owner_of(never_registered)


# ---------------------------------------------------------------------------
# Folding active contributions through the resolved hook
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn, fields, expected",
    [
        (lambda deltaT: deltaT, {"deltaT": 0.2}, 0.2),
        (lambda deltaT: deltaT * 0.5, {"deltaT": 4.0}, 2.0),  # resolves field from ctx
        (
            lambda self, deltaT: deltaT,
            {"deltaT": 0.2},
            0.2,
        ),  # method form: self skipped
    ],
)
def test_resolved_interface_folds_a_single_active_contribution(
    constraint: Any, fn: Callable[..., float], fields: dict[str, float], expected: float
) -> None:
    rt = contributor(constraint, fn, "courant")
    assert bound(constraint, [rt], fields)() == pytest.approx(expected)


def test_inactive_contributing_model_is_excluded_from_the_fold(constraint: Any) -> None:
    # Registered, but its runtime is NOT in the Context; its 0.001 would win the
    # min if it wrongly folded.
    contributor(constraint, lambda deltaT: deltaT * 0.001, "courant")
    assert bound(constraint, [], {"deltaT": 0.2})() == VGREAT


def test_an_unrelated_runtime_does_not_activate_a_contribution(constraint: Any) -> None:
    contributor(constraint, lambda deltaT: deltaT * 0.001, "courant")  # not in the ctx
    unrelated_rt = ModelRuntime(spec=Model("unrelated"), name="unrelated", config=None)
    assert bound(constraint, [unrelated_rt], {"deltaT": 0.2})() == VGREAT


def test_second_contributing_model_changes_the_fold_without_owner_edit(
    constraint: Any,
) -> None:
    courant_rt = contributor(constraint, lambda deltaT: deltaT, "courant")  # 0.2
    max_rt = contributor(constraint, lambda: 0.01, "maxDeltaT")  # smaller -> wins once active

    assert bound(constraint, [courant_rt], {"deltaT": 0.2})() == pytest.approx(0.2)
    assert bound(constraint, [courant_rt, max_rt], {"deltaT": 0.2})() == pytest.approx(0.01)


def test_contribution_resolves_config_param_from_its_own_runtime(
    constraint: Any,
) -> None:
    capped_rt = contributor(constraint, _cap_from_config, "capped", config=_CapConfig(value=0.05))
    assert bound(constraint, [capped_rt])() == pytest.approx(0.05)


@pytest.mark.parametrize(
    "fn, config, missing_param",
    [
        (_needs_deltaT, None, "deltaT"),  # field param absent from the Context
        (_cap_from_config, None, "cfg"),  # config param but the runtime has no config
    ],
)
def test_missing_contribution_parameter_error_names_interface_contribution_param(
    constraint: Any, fn: Callable[..., float], config: Any, missing_param: str
) -> None:
    rt = contributor(constraint, fn, "courant", config=config)
    with pytest.raises(ValueError) as excinfo:
        bound(constraint, [rt])()
    message = str(excinfo.value)
    assert constraint.name in message
    assert fn.__name__ in message
    assert missing_param in message


def test_two_cases_fold_independently_without_leak(constraint: Any) -> None:
    courant_rt = contributor(constraint, lambda deltaT: deltaT, "courant")
    max_rt = contributor(constraint, lambda: 0.01, "maxDeltaT")

    # Case A: only courant active (0.2); maxDeltaT would give 0.01 < 0.2 if it leaked.
    bound_a = bound(constraint, [courant_rt], {"deltaT": 0.2})
    # Case B: only maxDeltaT active (0.01); courant would give 0.005 < 0.01 if it leaked.
    bound_b = bound(constraint, [max_rt], {"deltaT": 0.005})

    assert bound_a() == pytest.approx(0.2)
    assert bound_b() == pytest.approx(0.01)
    assert bound_a() == pytest.approx(0.2)  # re-call after B: no leak


def test_interface_fold_is_agnostic_to_the_combine_function() -> None:
    accumulator = Model("accumulator")

    @accumulator.interface
    def sourceTerms(parts: Iterable[float]) -> float:
        return sum(parts)

    rate_a = Model("rateA")
    rate_b = Model("rateB")

    @rate_a.contributes(sourceTerms)
    def part_a() -> float:
        return 2.0

    @rate_b.contributes(sourceTerms)
    def part_b() -> float:
        return 3.0

    a_rt = ModelRuntime(spec=rate_a, name="rateA", config=None)
    b_rt = ModelRuntime(spec=rate_b, name="rateB", config=None)
    assert bound(sourceTerms, [a_rt, b_rt])() == pytest.approx(5.0)


def test_two_same_named_contributions_both_fold() -> None:
    accumulator = Model("accumulator")

    @accumulator.interface
    def sourceTerms(parts: Iterable[float]) -> float:
        return sum(parts)

    model_a = Model("modelA")
    model_b = Model("modelB")

    @model_a.contributes(sourceTerms)
    def rate() -> float:
        return 2.0

    @model_b.contributes(sourceTerms)
    def rate() -> float:  # noqa: F811  - same name, distinct function object
        return 3.0

    assert len(sourceTerms.contributions) == 2
    a_rt = ModelRuntime(spec=model_a, name="modelA", config=None)
    b_rt = ModelRuntime(spec=model_b, name="modelB", config=None)
    assert bound(sourceTerms, [a_rt, b_rt])() == pytest.approx(5.0)


def test_contribution_resolves_a_depends_marked_parameter(constraint: Any) -> None:
    def provide_limit() -> float:
        return 0.25

    def courant_limit(limit: Annotated[float, Depends(provide_limit)]) -> float:
        return limit

    rt = contributor(constraint, courant_limit, "courant")
    assert bound(constraint, [rt])() == pytest.approx(0.25)


def test_defaulted_contribution_parameter_still_requires_a_provider(
    constraint: Any,
) -> None:
    rt = contributor(constraint, lambda deltaT=1.0: deltaT, "courant")
    with pytest.raises(ValueError, match="no provider supplies it"):
        bound(constraint, [rt])()


def test_a_resolved_interface_reads_fields_at_call_time(constraint: Any) -> None:
    # The injection binds the hook to the live Context object; fields published
    # AFTER the injection (set_time_step writes ctx.fields["deltaT"] mid-body)
    # are what the contributions resolve against at call time.
    rt = contributor(constraint, lambda deltaT: deltaT, "capper")
    ctx = Context(fields={}, models={"capper": rt})
    resolved = constraint.resolve(ctx)

    with pytest.raises(ValueError, match="no provider supplies it"):
        resolved()  # deltaT not yet published
    ctx.fields["deltaT"] = 0.25
    assert resolved() == pytest.approx(0.25)


def test_contribution_from_a_foreign_model_family_still_folds(constraint: Any) -> None:
    # The contributor lives in a different model family than the interface owner.
    solver_side = Model("solverSideRule")

    @solver_side.contributes(constraint)
    def rule(deltaT: float) -> float:
        return deltaT * 0.5

    assert constraint.owner_of(rule) is solver_side
    solver_rt = ModelRuntime(spec=solver_side, name="solverSideRule", config=None)
    assert bound(constraint, [solver_rt], {"deltaT": 4.0})() == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Resolver injection of a hook-annotated parameter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn, active, expected",
    [
        (lambda deltaT: deltaT, True, 0.2),  # active contributor is injected and folds
        (lambda deltaT: deltaT * 0.001, False, VGREAT),  # none active -> default
    ],
)
def test_resolver_injects_the_live_bound_hook_for_an_interface_typed_param(
    constraint: Any, fn: Callable[..., float], active: bool, expected: float
) -> None:
    courant_rt = contributor(constraint, fn, "courant")
    ctx = Context(
        fields={"deltaT": 0.2},
        models={"courant": courant_rt} if active else {},
    )

    def set_time_step(self: Any, constraints: constraint) -> float:  # type: ignore[valid-type]
        return constraints()

    resolver = DependencyResolver()
    kwargs = resolver.resolve_arguments(set_time_step, ctx=ctx)
    assert callable(kwargs["constraints"])
    assert kwargs["constraints"]() == pytest.approx(expected)


def test_injection_needs_no_owner_runtime_in_the_context(constraint: Any) -> None:
    # Activation is decided by the CONTRIBUTING runtimes alone — nothing about
    # the owning model has to be registered for the interface to resolve.
    rt = contributor(constraint, lambda deltaT: deltaT, "courant")
    ctx = Context(fields={"deltaT": 0.2}, models={"courant": rt})

    def set_time_step(self: Any, constraints: constraint) -> float:  # type: ignore[valid-type]
        return constraints()

    kwargs = DependencyResolver().resolve_arguments(set_time_step, ctx=ctx)
    assert kwargs["constraints"]() == pytest.approx(0.2)


def test_resolver_raises_for_an_interface_param_without_a_context(constraint: Any) -> None:
    def set_time_step(self: Any, constraints: constraint) -> float:  # type: ignore[valid-type]
        return constraints()

    with pytest.raises(ValueError, match="no Context"):
        DependencyResolver().resolve_arguments(set_time_step, ctx=None)
