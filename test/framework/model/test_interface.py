# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for model-owned interfaces: declaring an interface on a Model, its
empty-fold default, owner reachability, and contribution registration."""

# NOTE: no `from __future__ import annotations` — a ModelInterface handle is meant
# to be used as a live operation-parameter annotation in a later iteration; keep
# these modules PEP 563-free so annotations stay non-string objects.

from typing import Annotated, Any, Callable, Iterable

import pytest

from neofoam.framework.context import Context
from neofoam.framework.dependency_resolver import DependencyResolver
from neofoam.framework.initialization.depends import Depends
from neofoam.framework.model import (
    BoundModelInterface,
    Model,
    ModelInterface,
    active_contributors,
    bind_model_interface,
    bind_owned_interfaces,
)
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
    """A model-owned ``timeStepConstraint`` interface (min-fold, VGREAT default).

    The owning ``Model`` is reachable via ``constraint.owner``.
    """
    return _loop_with_constraint()


def contributor(
    interface: Any, fn: Callable[..., Any], name: str, config: Any = None
) -> ModelRuntime:
    """A fresh ``Model`` named *name* contributing *fn* to *interface*, returned
    as the live ``ModelRuntime`` the bound interface folds over."""
    model = Model(name)
    model.contributes(interface)(fn)
    return ModelRuntime(spec=model, name=name, config=config)


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


def test_interface_decorator_returns_handle_named_after_the_fold(
    constraint: Any,
) -> None:
    assert isinstance(constraint, ModelInterface)
    assert constraint.name == "timeStepConstraint"


def test_handle_round_trips_through_its_owning_model(constraint: Any) -> None:
    owner = constraint.owner
    assert owner.name == "solutionLoop"
    assert owner.declared_interfaces["timeStepConstraint"] is constraint


def test_redeclaring_the_same_interface_name_is_rejected(constraint: Any) -> None:
    with pytest.raises(RuntimeError, match="already declared"):

        @constraint.owner.interface
        def timeStepConstraint(limits: Iterable[float]) -> float:  # noqa: F811
            return min(limits, default=VGREAT)


def test_same_interface_name_on_two_models_is_allowed() -> None:
    loop_a = Model("loopA")
    loop_b = Model("loopB")

    @loop_a.interface
    def timeStepConstraint(limits: Iterable[float]) -> float:
        return min(limits, default=VGREAT)

    @loop_b.interface
    def timeStepConstraint(limits: Iterable[float]) -> float:  # noqa: F811
        return min(limits, default=VGREAT)

    assert loop_a.declared_interfaces["timeStepConstraint"].owner is loop_a
    assert loop_b.declared_interfaces["timeStepConstraint"].owner is loop_b


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
def test_fold_reduces_inputs_via_the_declared_combine(
    constraint: Any, limits: Iterable[float], expected: float
) -> None:
    assert constraint.fold(limits) == expected


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
    # Registration is recorded for later gathering but is inert here: the empty
    # fold still returns the default even after a contribution is registered.
    courant = Model("courant")

    @courant.contributes(constraint)
    def courant_limit(deltaT: float) -> float:
        return 0.001

    assert constraint.fold([]) == VGREAT


def test_contributes_against_a_non_interface_target_raises() -> None:
    courant = Model("courant")
    with pytest.raises(TypeError, match="must be a ModelInterface"):

        @courant.contributes(lambda limits: min(limits))  # type: ignore[arg-type]
        def bad(deltaT: float) -> float:
            return deltaT


def test_owner_of_unregistered_function_raises(constraint: Any) -> None:
    def never_registered(deltaT: float) -> float:
        return deltaT

    with pytest.raises(KeyError, match="not a registered contribution"):
        constraint.owner_of(never_registered)


# ---------------------------------------------------------------------------
# Folding active contributions through a BoundModelInterface
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
def test_bound_interface_folds_a_single_active_contribution(
    constraint: Any, fn: Callable[..., float], fields: dict[str, float], expected: float
) -> None:
    rt = contributor(constraint, fn, "courant")
    bound = BoundModelInterface(constraint, [rt], Context(fields=fields, models={}))
    assert bound() == pytest.approx(expected)


def test_inactive_contributing_model_is_excluded_from_the_fold(constraint: Any) -> None:
    # Registered, but its runtime is NOT among the active contributors for this
    # case; its 0.001 would win the min if it wrongly folded.
    contributor(constraint, lambda deltaT: deltaT * 0.001, "courant")
    bound = BoundModelInterface(
        constraint, [], Context(fields={"deltaT": 0.2}, models={})
    )
    assert bound() == VGREAT


def test_second_contributing_model_changes_the_fold_without_owner_edit(
    constraint: Any,
) -> None:
    courant_rt = contributor(constraint, lambda deltaT: deltaT, "courant")  # 0.2
    max_rt = contributor(
        constraint, lambda: 0.01, "maxDeltaT"
    )  # smaller -> wins once active
    ctx = Context(fields={"deltaT": 0.2}, models={})

    only_courant = BoundModelInterface(constraint, [courant_rt], ctx)
    assert only_courant() == pytest.approx(0.2)

    both = BoundModelInterface(constraint, [courant_rt, max_rt], ctx)
    assert both() == pytest.approx(0.01)


def test_contribution_resolves_config_param_from_its_own_runtime(
    constraint: Any,
) -> None:
    capped_rt = contributor(
        constraint, _cap_from_config, "capped", config=_CapConfig(value=0.05)
    )
    bound = BoundModelInterface(constraint, [capped_rt], Context(fields={}, models={}))
    assert bound() == pytest.approx(0.05)


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
    bound = BoundModelInterface(constraint, [rt], Context(fields={}, models={}))
    with pytest.raises(ValueError) as excinfo:
        bound()
    message = str(excinfo.value)
    assert constraint.name in message
    assert fn.__name__ in message
    assert missing_param in message


def _direct_bind(interface: Any, active: Any, ctx: Any) -> Any:
    return BoundModelInterface(interface, active, ctx)


def _autowired_bind(interface: Any, active: Any, ctx: Any) -> Any:
    owner_rt = ModelRuntime(spec=interface.owner, name="solutionLoop", config=None)
    bind_owned_interfaces(owner_rt, active, ctx)
    return owner_rt.bound_interfaces["timeStepConstraint"]


@pytest.mark.parametrize("make_bound", [_direct_bind, _autowired_bind])
def test_two_cases_fold_independently_without_leak(
    constraint: Any, make_bound: Callable[..., Any]
) -> None:
    courant_rt = contributor(constraint, lambda deltaT: deltaT, "courant")
    max_rt = contributor(constraint, lambda: 0.01, "maxDeltaT")

    # Case A: only courant active (0.2); maxDeltaT would give 0.01 < 0.2 if it leaked.
    bound_a = make_bound(
        constraint, [courant_rt], Context(fields={"deltaT": 0.2}, models={})
    )
    # Case B: only maxDeltaT active (0.01); courant would give 0.005 < 0.01 if it leaked.
    bound_b = make_bound(
        constraint, [max_rt], Context(fields={"deltaT": 0.005}, models={})
    )

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
    bound = BoundModelInterface(
        sourceTerms, [a_rt, b_rt], Context(fields={}, models={})
    )
    assert bound() == pytest.approx(5.0)


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
    bound = BoundModelInterface(
        sourceTerms, [a_rt, b_rt], Context(fields={}, models={})
    )
    assert bound() == pytest.approx(5.0)


def test_contribution_resolves_a_depends_marked_parameter(constraint: Any) -> None:
    def provide_limit() -> float:
        return 0.25

    def courant_limit(limit: Annotated[float, Depends(provide_limit)]) -> float:
        return limit

    rt = contributor(constraint, courant_limit, "courant")
    bound = BoundModelInterface(constraint, [rt], Context(fields={}, models={}))
    assert bound() == pytest.approx(0.25)


def test_defaulted_contribution_parameter_still_requires_a_provider(
    constraint: Any,
) -> None:
    rt = contributor(constraint, lambda deltaT=1.0: deltaT, "courant")
    bound = BoundModelInterface(constraint, [rt], Context(fields={}, models={}))
    with pytest.raises(ValueError, match="no provider supplies it"):
        bound()


def test_bound_interface_folds_against_the_call_time_context(constraint: Any) -> None:
    # The handle is bound with an EMPTY Context (no live capture — the GC-safe
    # contract); the live Context is supplied at CALL time and is what the
    # contributions resolve against.
    rt = contributor(constraint, lambda deltaT: deltaT, "capper")
    bound = BoundModelInterface(constraint, [rt], Context(fields={}, models={}))

    live = Context(fields={"deltaT": 0.25}, models={})
    assert bound(live) == pytest.approx(0.25)  # folds against the passed ctx
    with pytest.raises(ValueError, match="no provider supplies it"):
        bound()  # empty bound ctx -> unresolved


def test_contribution_from_a_foreign_model_family_still_folds(constraint: Any) -> None:
    # The contributor lives in a different model family than the interface owner.
    solver_side = Model("solverSideRule")

    @solver_side.contributes(constraint)
    def rule(deltaT: float) -> float:
        return deltaT * 0.5

    assert constraint.owner_of(rule) is solver_side
    solver_rt = ModelRuntime(spec=solver_side, name="solverSideRule", config=None)
    bound = BoundModelInterface(
        constraint, [solver_rt], Context(fields={"deltaT": 4.0}, models={})
    )
    assert bound() == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Binding an interface onto its owning runtime + resolver injection
# ---------------------------------------------------------------------------


def test_bound_interface_lives_on_the_owning_runtime(constraint: Any) -> None:
    courant_rt = contributor(constraint, lambda deltaT: deltaT, "courant")
    loop_rt = ModelRuntime(spec=constraint.owner, name="solutionLoop", config=None)
    ctx = Context(fields={"deltaT": 0.2}, models={})

    bound = bind_model_interface(loop_rt, constraint, [courant_rt], ctx)
    assert loop_rt.bound_interfaces["timeStepConstraint"] is bound
    assert bound() == pytest.approx(0.2)


def test_active_contributors_filters_by_spec_identity(constraint: Any) -> None:
    courant_rt = contributor(constraint, lambda deltaT: deltaT, "courant")
    unrelated_rt = ModelRuntime(spec=Model("unrelated"), name="unrelated", config=None)

    active = active_contributors(constraint, [courant_rt, unrelated_rt])
    assert active == [courant_rt]


@pytest.mark.parametrize(
    "fn, active, expected",
    [
        (lambda deltaT: deltaT, True, 0.2),  # active contributor is injected and folds
        (lambda deltaT: deltaT * 0.001, False, VGREAT),  # none active -> default
    ],
)
def test_resolver_injects_bound_interface_for_interface_typed_param(
    constraint: Any, fn: Callable[..., float], active: bool, expected: float
) -> None:
    loop_rt = ModelRuntime(spec=constraint.owner, name="solutionLoop", config=None)
    courant_rt = contributor(constraint, fn, "courant")
    ctx = Context(
        fields={"deltaT": 0.2},
        models={"solutionLoop": loop_rt, "courant": courant_rt},
    )
    bind_model_interface(loop_rt, constraint, [courant_rt] if active else [], ctx)

    def set_time_step(self: Any, constraints: constraint) -> float:  # type: ignore[valid-type]
        return constraints()

    resolver = DependencyResolver()
    kwargs = resolver.resolve_arguments(set_time_step, ctx=ctx)
    assert isinstance(kwargs["constraints"], BoundModelInterface)
    assert kwargs["constraints"]() == pytest.approx(expected)


@pytest.mark.parametrize(
    "make_ctx, match",
    [
        # owner runtime present, but its interface was never bound for this case
        (
            lambda c: Context(
                fields={},
                models={
                    "solutionLoop": ModelRuntime(
                        spec=c.owner, name="solutionLoop", config=None
                    )
                },
            ),
            "is not bound for this case",
        ),
        # no Context supplied at all
        (lambda c: None, "no Context"),
        # owner runtime absent from ctx.models entirely
        (lambda c: Context(fields={}, models={}), "is not bound for this case"),
    ],
)
def test_resolver_raises_for_unbound_interface_param(
    constraint: Any, make_ctx: Callable[[Any], Any], match: str
) -> None:
    def set_time_step(self: Any, constraints: constraint) -> float:  # type: ignore[valid-type]
        return constraints()

    resolver = DependencyResolver()
    with pytest.raises(ValueError, match=match):
        resolver.resolve_arguments(set_time_step, ctx=make_ctx(constraint))


@pytest.mark.parametrize(
    "candidate_contributes, expected",
    [
        (True, 0.2),  # the candidate contributes -> its limit folds in
        (False, VGREAT),  # candidate contributes nothing -> excluded -> default
    ],
)
def test_auto_wiring_binds_only_active_contributors(
    constraint: Any, candidate_contributes: bool, expected: float
) -> None:
    loop_rt = ModelRuntime(spec=constraint.owner, name="solutionLoop", config=None)
    if candidate_contributes:
        candidate = contributor(constraint, lambda deltaT: deltaT, "courant")
    else:
        # courant's 0.001 would win the min if it leaked, but it is NOT a candidate.
        contributor(constraint, lambda deltaT: deltaT * 0.001, "courant")
        candidate = ModelRuntime(spec=Model("unrelated"), name="unrelated", config=None)

    ctx = Context(fields={"deltaT": 0.2}, models={})
    bind_owned_interfaces(loop_rt, [candidate], ctx)

    bound = loop_rt.bound_interfaces["timeStepConstraint"]
    assert isinstance(bound, BoundModelInterface)
    assert bound() == pytest.approx(expected)
