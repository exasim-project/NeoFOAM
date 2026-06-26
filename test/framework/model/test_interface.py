# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for model-owned interfaces: declaring an interface on a Model, its
empty-fold default, owner reachability, and contribution registration."""

# NOTE: no `from __future__ import annotations` — a ModelInterface handle is meant
# to be used as a live operation-parameter annotation in a later iteration; keep
# these modules PEP 563-free so annotations stay non-string objects.

from typing import Annotated, Any, Iterable

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


def test_interface_decorator_returns_handle_named_after_the_fold() -> None:
    loop = Model("solutionLoop")

    @loop.interface
    def timeStepConstraint(limits: Iterable[float]) -> float:
        return min(limits, default=VGREAT)

    assert isinstance(timeStepConstraint, ModelInterface)
    assert timeStepConstraint.name == "timeStepConstraint"


def test_empty_fold_returns_the_documented_default() -> None:
    loop = Model("solutionLoop")

    @loop.interface
    def timeStepConstraint(limits: Iterable[float]) -> float:
        return min(limits, default=VGREAT)

    assert timeStepConstraint.fold([]) == VGREAT


def test_fold_combines_supplied_values() -> None:
    loop = Model("solutionLoop")

    @loop.interface
    def timeStepConstraint(limits: Iterable[float]) -> float:
        return min(limits, default=VGREAT)

    assert timeStepConstraint.fold([2.0, 1.0, 3.0]) == 1.0


def test_handle_is_owned_by_its_declaring_model() -> None:
    loop = Model("solutionLoop")

    @loop.interface
    def timeStepConstraint(limits: Iterable[float]) -> float:
        return min(limits, default=VGREAT)

    assert timeStepConstraint.owner is loop


def test_handle_is_reachable_from_the_owning_model() -> None:
    loop = Model("solutionLoop")

    @loop.interface
    def timeStepConstraint(limits: Iterable[float]) -> float:
        return min(limits, default=VGREAT)

    assert loop.declared_interfaces["timeStepConstraint"] is timeStepConstraint


def test_redeclaring_the_same_interface_name_is_rejected() -> None:
    loop = Model("solutionLoop")

    @loop.interface
    def timeStepConstraint(limits: Iterable[float]) -> float:
        return min(limits, default=VGREAT)

    with pytest.raises(RuntimeError, match="already declared"):

        @loop.interface
        def timeStepConstraint(limits: Iterable[float]) -> float:  # noqa: F811
            return min(limits, default=VGREAT)


def test_contributes_registers_the_function_against_the_interface() -> None:
    loop = Model("solutionLoop")

    @loop.interface
    def timeStepConstraint(limits: Iterable[float]) -> float:
        return min(limits, default=VGREAT)

    courant = Model("courant")

    @courant.contributes(timeStepConstraint)
    def courant_limit(deltaT: float) -> float:
        return deltaT * 0.5

    assert courant_limit in timeStepConstraint.contributions


def test_contributes_records_the_contributing_model_as_owner() -> None:
    loop = Model("solutionLoop")

    @loop.interface
    def timeStepConstraint(limits: Iterable[float]) -> float:
        return min(limits, default=VGREAT)

    courant = Model("courant")

    @courant.contributes(timeStepConstraint)
    def courant_limit(deltaT: float) -> float:
        return deltaT * 0.5

    assert timeStepConstraint.owner_of(courant_limit) is courant


def test_contributes_returns_the_function_unchanged() -> None:
    loop = Model("solutionLoop")

    @loop.interface
    def timeStepConstraint(limits: Iterable[float]) -> float:
        return min(limits, default=VGREAT)

    courant = Model("courant")

    def courant_limit(deltaT: float) -> float:
        return deltaT * 0.5

    returned = courant.contributes(timeStepConstraint)(courant_limit)
    assert returned is courant_limit


def test_two_models_contribute_independently_to_one_interface() -> None:
    loop = Model("solutionLoop")

    @loop.interface
    def timeStepConstraint(limits: Iterable[float]) -> float:
        return min(limits, default=VGREAT)

    courant = Model("courant")
    max_delta_t = Model("maxDeltaT")

    @courant.contributes(timeStepConstraint)
    def courant_limit(deltaT: float) -> float:
        return deltaT

    @max_delta_t.contributes(timeStepConstraint)
    def fixed_cap() -> float:
        return 0.01

    assert set(timeStepConstraint.contributions) == {courant_limit, fixed_cap}
    assert timeStepConstraint.owner_of(courant_limit) is courant
    assert timeStepConstraint.owner_of(fixed_cap) is max_delta_t


def test_registration_does_not_yet_participate_in_the_fold() -> None:
    # Registration is recorded for later gathering but is inert here: the empty
    # fold still returns the default even after a contribution is registered.
    loop = Model("solutionLoop")

    @loop.interface
    def timeStepConstraint(limits: Iterable[float]) -> float:
        return min(limits, default=VGREAT)

    courant = Model("courant")

    @courant.contributes(timeStepConstraint)
    def courant_limit(deltaT: float) -> float:
        return 0.001

    assert timeStepConstraint.fold([]) == VGREAT


def test_contributions_are_returned_in_registration_order() -> None:
    loop = Model("solutionLoop")

    @loop.interface
    def timeStepConstraint(limits: Iterable[float]) -> float:
        return min(limits, default=VGREAT)

    courant = Model("courant")
    max_delta_t = Model("maxDeltaT")

    @courant.contributes(timeStepConstraint)
    def courant_limit(deltaT: float) -> float:
        return deltaT

    @max_delta_t.contributes(timeStepConstraint)
    def fixed_cap() -> float:
        return 0.01

    assert timeStepConstraint.contributions == (courant_limit, fixed_cap)


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


def test_contributes_against_a_non_interface_target_raises() -> None:
    courant = Model("courant")
    with pytest.raises(TypeError, match="must be a ModelInterface"):

        @courant.contributes(lambda limits: min(limits))  # type: ignore[arg-type]
        def bad(deltaT: float) -> float:
            return deltaT


def test_owner_of_unregistered_function_raises() -> None:
    loop = Model("solutionLoop")

    @loop.interface
    def timeStepConstraint(limits: Iterable[float]) -> float:
        return min(limits, default=VGREAT)

    def never_registered(deltaT: float) -> float:
        return deltaT

    with pytest.raises(KeyError, match="not a registered contribution"):
        timeStepConstraint.owner_of(never_registered)


def test_fold_handles_single_duplicate_and_generator_inputs() -> None:
    loop = Model("solutionLoop")

    @loop.interface
    def timeStepConstraint(limits: Iterable[float]) -> float:
        return min(limits, default=VGREAT)

    assert timeStepConstraint.fold([5.0]) == 5.0
    assert timeStepConstraint.fold([1.0, 1.0]) == 1.0
    assert timeStepConstraint.fold(x for x in (3.0, 2.0, 4.0)) == 2.0


class _CapConfig(BaseConfig):
    value: float


def _loop_with_constraint() -> Any:
    loop = Model("solutionLoop")

    @loop.interface
    def timeStepConstraint(limits: Iterable[float]) -> float:
        return min(limits, default=VGREAT)

    return timeStepConstraint


def test_active_contributing_runtime_folds() -> None:
    timeStepConstraint = _loop_with_constraint()
    courant = Model("courant")

    @courant.contributes(timeStepConstraint)
    def courant_limit(deltaT: float) -> float:
        return deltaT

    courant_rt = ModelRuntime(spec=courant, name="courant", config=None)
    ctx = Context(fields={"deltaT": 0.2}, models={})
    bound = BoundModelInterface(timeStepConstraint, [courant_rt], ctx)
    assert bound() == pytest.approx(0.2)


def test_inactive_contributing_model_is_excluded_from_the_fold() -> None:
    timeStepConstraint = _loop_with_constraint()
    courant = Model("courant")

    @courant.contributes(timeStepConstraint)
    def courant_limit(deltaT: float) -> float:
        return deltaT * 0.001  # would win the min if it wrongly folded

    ctx = Context(fields={"deltaT": 0.2}, models={})
    # courant runtime NOT among the active contributors for this case.
    bound = BoundModelInterface(timeStepConstraint, [], ctx)
    assert bound() == VGREAT


def test_second_contributing_model_changes_the_fold_without_owner_edit() -> None:
    timeStepConstraint = _loop_with_constraint()
    courant = Model("courant")
    max_delta_t = Model("maxDeltaT")

    @courant.contributes(timeStepConstraint)
    def courant_limit(deltaT: float) -> float:
        return deltaT  # 0.2

    @max_delta_t.contributes(timeStepConstraint)
    def fixed_cap() -> float:
        return 0.01  # smaller -> wins the min once its model is active

    ctx = Context(fields={"deltaT": 0.2}, models={})
    courant_rt = ModelRuntime(spec=courant, name="courant", config=None)
    max_rt = ModelRuntime(spec=max_delta_t, name="maxDeltaT", config=None)

    only_courant = BoundModelInterface(timeStepConstraint, [courant_rt], ctx)
    assert only_courant() == pytest.approx(0.2)

    both = BoundModelInterface(timeStepConstraint, [courant_rt, max_rt], ctx)
    assert both() == pytest.approx(0.01)


def test_contribution_resolves_field_param_from_context() -> None:
    timeStepConstraint = _loop_with_constraint()
    courant = Model("courant")

    @courant.contributes(timeStepConstraint)
    def courant_limit(deltaT: float) -> float:
        return deltaT * 0.5

    courant_rt = ModelRuntime(spec=courant, name="courant", config=None)
    ctx = Context(fields={"deltaT": 4.0}, models={})
    bound = BoundModelInterface(timeStepConstraint, [courant_rt], ctx)
    assert bound() == pytest.approx(2.0)


def test_contribution_resolves_config_param_from_its_own_runtime() -> None:
    timeStepConstraint = _loop_with_constraint()
    capped = Model("capped")

    @capped.contributes(timeStepConstraint)
    def cap(cfg: _CapConfig) -> float:
        return cfg.value

    capped_rt = ModelRuntime(spec=capped, name="capped", config=_CapConfig(value=0.05))
    ctx = Context(fields={}, models={})
    bound = BoundModelInterface(timeStepConstraint, [capped_rt], ctx)
    assert bound() == pytest.approx(0.05)


def test_missing_contribution_parameter_raises_naming_interface_contribution_and_param() -> (
    None
):
    timeStepConstraint = _loop_with_constraint()
    courant = Model("courant")

    @courant.contributes(timeStepConstraint)
    def courant_limit(deltaT: float) -> float:
        return deltaT

    courant_rt = ModelRuntime(spec=courant, name="courant", config=None)
    ctx = Context(fields={}, models={})  # 'deltaT' absent
    bound = BoundModelInterface(timeStepConstraint, [courant_rt], ctx)
    with pytest.raises(ValueError) as excinfo:
        bound()
    message = str(excinfo.value)
    assert "timeStepConstraint" in message
    assert "courant_limit" in message
    assert "deltaT" in message


def test_two_cases_fold_independently_without_leak() -> None:
    timeStepConstraint = _loop_with_constraint()
    courant = Model("courant")
    max_delta_t = Model("maxDeltaT")

    @courant.contributes(timeStepConstraint)
    def courant_limit(deltaT: float) -> float:
        return deltaT

    @max_delta_t.contributes(timeStepConstraint)
    def fixed_cap() -> float:
        return 0.01

    courant_rt = ModelRuntime(spec=courant, name="courant", config=None)
    max_rt = ModelRuntime(spec=max_delta_t, name="maxDeltaT", config=None)

    # Case A: only courant active (0.2); maxDeltaT would give 0.01 < 0.2 if it leaked.
    ctx_a = Context(fields={"deltaT": 0.2}, models={})
    bound_a = BoundModelInterface(timeStepConstraint, [courant_rt], ctx_a)
    # Case B: only maxDeltaT active (0.01); courant would give 0.005 < 0.01 if it leaked.
    ctx_b = Context(fields={"deltaT": 0.005}, models={})
    bound_b = BoundModelInterface(timeStepConstraint, [max_rt], ctx_b)

    assert bound_a() == pytest.approx(0.2)
    assert bound_b() == pytest.approx(0.01)
    assert bound_a() == pytest.approx(0.2)  # re-call after B: no leak


def test_bound_interface_lives_on_the_owning_runtime() -> None:
    timeStepConstraint = _loop_with_constraint()
    courant = Model("courant")

    @courant.contributes(timeStepConstraint)
    def courant_limit(deltaT: float) -> float:
        return deltaT

    loop_rt = ModelRuntime(
        spec=timeStepConstraint.owner, name="solutionLoop", config=None
    )
    courant_rt = ModelRuntime(spec=courant, name="courant", config=None)
    ctx = Context(fields={"deltaT": 0.2}, models={})

    bound = bind_model_interface(loop_rt, timeStepConstraint, [courant_rt], ctx)
    assert loop_rt.bound_interfaces["timeStepConstraint"] is bound
    assert bound() == pytest.approx(0.2)


def test_resolver_injects_bound_interface_for_interface_typed_param() -> None:
    timeStepConstraint = _loop_with_constraint()
    courant = Model("courant")

    @courant.contributes(timeStepConstraint)
    def courant_limit(deltaT: float) -> float:
        return deltaT

    loop_rt = ModelRuntime(
        spec=timeStepConstraint.owner, name="solutionLoop", config=None
    )
    courant_rt = ModelRuntime(spec=courant, name="courant", config=None)
    ctx = Context(
        fields={"deltaT": 0.2},
        models={"solutionLoop": loop_rt, "courant": courant_rt},
    )
    bind_model_interface(loop_rt, timeStepConstraint, [courant_rt], ctx)

    def set_time_step(self: Any, constraints: timeStepConstraint) -> float:  # type: ignore[valid-type]
        return constraints()

    resolver = DependencyResolver()
    kwargs = resolver.resolve_arguments(set_time_step, ctx=ctx)
    assert isinstance(kwargs["constraints"], BoundModelInterface)


def test_bound_interface_call_folds_active_contributions() -> None:
    timeStepConstraint = _loop_with_constraint()
    courant = Model("courant")

    @courant.contributes(timeStepConstraint)
    def courant_limit(deltaT: float) -> float:
        return deltaT

    loop_rt = ModelRuntime(
        spec=timeStepConstraint.owner, name="solutionLoop", config=None
    )
    courant_rt = ModelRuntime(spec=courant, name="courant", config=None)
    ctx = Context(
        fields={"deltaT": 0.2},
        models={"solutionLoop": loop_rt, "courant": courant_rt},
    )
    bind_model_interface(loop_rt, timeStepConstraint, [courant_rt], ctx)

    def set_time_step(self: Any, constraints: timeStepConstraint) -> float:  # type: ignore[valid-type]
        return constraints()

    resolver = DependencyResolver()
    kwargs = resolver.resolve_arguments(set_time_step, ctx=ctx)
    assert kwargs["constraints"]() == pytest.approx(0.2)


def test_bound_interface_with_no_active_contributors_returns_default() -> None:
    timeStepConstraint = _loop_with_constraint()
    courant = Model("courant")

    @courant.contributes(timeStepConstraint)
    def courant_limit(deltaT: float) -> float:
        return deltaT * 0.001

    loop_rt = ModelRuntime(
        spec=timeStepConstraint.owner, name="solutionLoop", config=None
    )
    ctx = Context(fields={"deltaT": 0.2}, models={"solutionLoop": loop_rt})
    bind_model_interface(loop_rt, timeStepConstraint, [], ctx)

    def set_time_step(self: Any, constraints: timeStepConstraint) -> float:  # type: ignore[valid-type]
        return constraints()

    resolver = DependencyResolver()
    kwargs = resolver.resolve_arguments(set_time_step, ctx=ctx)
    assert kwargs["constraints"]() == VGREAT


def test_resolver_raises_when_interface_is_not_bound_for_the_case() -> None:
    timeStepConstraint = _loop_with_constraint()
    loop_rt = ModelRuntime(
        spec=timeStepConstraint.owner, name="solutionLoop", config=None
    )
    ctx = Context(fields={}, models={"solutionLoop": loop_rt})  # not bound

    def set_time_step(self: Any, constraints: timeStepConstraint) -> float:  # type: ignore[valid-type]
        return constraints()

    resolver = DependencyResolver()
    with pytest.raises(ValueError, match="is not bound for this case"):
        resolver.resolve_arguments(set_time_step, ctx=ctx)


def test_missing_config_parameter_raises_naming_interface_contribution_and_param() -> (
    None
):
    timeStepConstraint = _loop_with_constraint()
    capped = Model("capped")

    @capped.contributes(timeStepConstraint)
    def cap(cfg: _CapConfig) -> float:
        return cfg.value

    capped_rt = ModelRuntime(spec=capped, name="capped", config=None)  # no config
    ctx = Context(fields={}, models={})
    bound = BoundModelInterface(timeStepConstraint, [capped_rt], ctx)
    with pytest.raises(ValueError) as excinfo:
        bound()
    message = str(excinfo.value)
    assert "timeStepConstraint" in message
    assert "cap" in message
    assert "cfg" in message


def test_resolver_raises_for_interface_param_without_context() -> None:
    timeStepConstraint = _loop_with_constraint()

    def set_time_step(self: Any, constraints: timeStepConstraint) -> float:  # type: ignore[valid-type]
        return constraints()

    resolver = DependencyResolver()
    with pytest.raises(ValueError, match="no Context"):
        resolver.resolve_arguments(set_time_step, ctx=None)


def test_operation_method_contribution_folds_with_receiver_skipped() -> None:
    timeStepConstraint = _loop_with_constraint()
    courant = Model("courant")

    @courant.contributes(timeStepConstraint)
    def courant_limit(self: Any, deltaT: float) -> float:
        return deltaT  # `self` is the receiver, skipped by resolution

    courant_rt = ModelRuntime(spec=courant, name="courant", config=None)
    ctx = Context(fields={"deltaT": 0.2}, models={})
    bound = BoundModelInterface(timeStepConstraint, [courant_rt], ctx)
    assert bound() == pytest.approx(0.2)


def test_resolver_raises_when_owner_is_absent_from_context_models() -> None:
    timeStepConstraint = _loop_with_constraint()
    ctx = Context(fields={}, models={})  # owner runtime not registered at all

    def set_time_step(self: Any, constraints: timeStepConstraint) -> float:  # type: ignore[valid-type]
        return constraints()

    resolver = DependencyResolver()
    with pytest.raises(ValueError, match="is not bound for this case"):
        resolver.resolve_arguments(set_time_step, ctx=ctx)


def test_active_contributors_filters_by_spec_identity() -> None:
    timeStepConstraint = _loop_with_constraint()
    courant = Model("courant")
    unrelated = Model("unrelated")

    @courant.contributes(timeStepConstraint)
    def courant_limit(deltaT: float) -> float:
        return deltaT

    courant_rt = ModelRuntime(spec=courant, name="courant", config=None)
    unrelated_rt = ModelRuntime(spec=unrelated, name="unrelated", config=None)

    active = active_contributors(timeStepConstraint, [courant_rt, unrelated_rt])
    assert active == [courant_rt]


def test_auto_wiring_binds_active_contributors_per_case() -> None:
    timeStepConstraint = _loop_with_constraint()
    courant = Model("courant")

    @courant.contributes(timeStepConstraint)
    def courant_limit(deltaT: float) -> float:
        return deltaT

    loop_rt = ModelRuntime(
        spec=timeStepConstraint.owner, name="solutionLoop", config=None
    )
    courant_rt = ModelRuntime(spec=courant, name="courant", config=None)
    ctx = Context(fields={"deltaT": 0.2}, models={})

    bind_owned_interfaces(loop_rt, [courant_rt], ctx)
    bound = loop_rt.bound_interfaces["timeStepConstraint"]
    assert isinstance(bound, BoundModelInterface)
    assert bound() == pytest.approx(0.2)


def test_auto_wiring_excludes_a_non_contributing_candidate() -> None:
    timeStepConstraint = _loop_with_constraint()
    courant = Model("courant")
    unrelated = Model("unrelated")  # contributes to nothing

    @courant.contributes(timeStepConstraint)
    def courant_limit(deltaT: float) -> float:
        return deltaT * 0.001  # would win the min if it leaked in

    loop_rt = ModelRuntime(
        spec=timeStepConstraint.owner, name="solutionLoop", config=None
    )
    unrelated_rt = ModelRuntime(spec=unrelated, name="unrelated", config=None)
    ctx = Context(fields={"deltaT": 0.2}, models={})

    bind_owned_interfaces(loop_rt, [unrelated_rt], ctx)
    assert loop_rt.bound_interfaces["timeStepConstraint"]() == VGREAT


def test_auto_wiring_two_cases_do_not_leak() -> None:
    timeStepConstraint = _loop_with_constraint()
    courant = Model("courant")
    max_delta_t = Model("maxDeltaT")

    @courant.contributes(timeStepConstraint)
    def courant_limit(deltaT: float) -> float:
        return deltaT

    @max_delta_t.contributes(timeStepConstraint)
    def fixed_cap() -> float:
        return 0.01

    courant_rt = ModelRuntime(spec=courant, name="courant", config=None)
    max_rt = ModelRuntime(spec=max_delta_t, name="maxDeltaT", config=None)

    loop_a = ModelRuntime(
        spec=timeStepConstraint.owner, name="solutionLoop", config=None
    )
    ctx_a = Context(fields={"deltaT": 0.2}, models={})
    bind_owned_interfaces(loop_a, [courant_rt], ctx_a)  # only courant active

    loop_b = ModelRuntime(
        spec=timeStepConstraint.owner, name="solutionLoop", config=None
    )
    ctx_b = Context(fields={"deltaT": 0.005}, models={})
    bind_owned_interfaces(loop_b, [max_rt], ctx_b)  # only maxDeltaT active

    assert loop_a.bound_interfaces["timeStepConstraint"]() == pytest.approx(0.2)
    assert loop_b.bound_interfaces["timeStepConstraint"]() == pytest.approx(0.01)
    assert loop_a.bound_interfaces["timeStepConstraint"]() == pytest.approx(
        0.2
    )  # no leak


def test_interface_owner_and_contributor_live_in_different_families() -> None:
    core = Model("coreLoop")  # owner: a framework/core model

    @core.interface
    def timeStepConstraint(limits: Iterable[float]) -> float:
        return min(limits, default=VGREAT)

    solver_side = Model("solverSideRule")  # a model that would register elsewhere

    @solver_side.contributes(timeStepConstraint)
    def rule(deltaT: float) -> float:
        return deltaT * 0.5

    assert timeStepConstraint.owner is core
    assert timeStepConstraint.owner_of(rule) is solver_side

    solver_rt = ModelRuntime(spec=solver_side, name="solverSideRule", config=None)
    ctx = Context(fields={"deltaT": 4.0}, models={})
    bound = BoundModelInterface(timeStepConstraint, [solver_rt], ctx)
    assert bound() == pytest.approx(2.0)


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
    ctx = Context(fields={}, models={})
    bound = BoundModelInterface(sourceTerms, [a_rt, b_rt], ctx)
    assert bound() == pytest.approx(5.0)


def test_contribution_resolves_a_depends_marked_parameter() -> None:
    timeStepConstraint = _loop_with_constraint()
    courant = Model("courant")

    def provide_limit() -> float:
        return 0.25

    @courant.contributes(timeStepConstraint)
    def courant_limit(limit: Annotated[float, Depends(provide_limit)]) -> float:
        return limit

    courant_rt = ModelRuntime(spec=courant, name="courant", config=None)
    ctx = Context(fields={}, models={})
    bound = BoundModelInterface(timeStepConstraint, [courant_rt], ctx)
    assert bound() == pytest.approx(0.25)


def test_defaulted_contribution_parameter_still_requires_a_provider() -> None:
    timeStepConstraint = _loop_with_constraint()
    courant = Model("courant")

    @courant.contributes(timeStepConstraint)
    def courant_limit(deltaT: float = 1.0) -> float:
        return deltaT

    courant_rt = ModelRuntime(spec=courant, name="courant", config=None)
    ctx = Context(fields={}, models={})  # no 'deltaT' provider
    bound = BoundModelInterface(timeStepConstraint, [courant_rt], ctx)
    with pytest.raises(ValueError, match="no provider supplies it"):
        bound()


def test_bound_interface_folds_against_the_call_time_context() -> None:
    # The handle is bound with an EMPTY Context (no live capture — the GC-safe
    # contract); the live Context is supplied at CALL time and is what the
    # contributions resolve against.
    loop = Model("solutionLoop")

    @loop.interface
    def timeStepConstraint(limits: Iterable[float]) -> float:
        return min(limits, default=VGREAT)

    contributor = Model("capper")

    @contributor.contributes(timeStepConstraint)
    def cap(deltaT: float) -> float:
        return deltaT

    rt = ModelRuntime(spec=contributor, name="capper", config=None)
    bound = BoundModelInterface(timeStepConstraint, [rt], Context(fields={}, models={}))

    live = Context(fields={"deltaT": 0.25}, models={})
    assert bound(live) == pytest.approx(0.25)  # folds against the passed ctx
    with pytest.raises(ValueError, match="no provider supplies it"):
        bound()  # empty bound ctx -> unresolved


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
    ctx = Context(fields={}, models={})
    bound = BoundModelInterface(sourceTerms, [a_rt, b_rt], ctx)
    assert bound() == pytest.approx(5.0)
