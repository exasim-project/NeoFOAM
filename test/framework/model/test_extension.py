# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for operation-declared extensions: declaring hooks via
``@<extension>.defines``, contributing via ``@<model>.contributes(<hook>)``, and
dispatching calls on the injected ``BoundExtension`` — the declaration body
combines the contribution results (or a broadcast hook returns them raw)."""

# NOTE: no `from __future__ import annotations` — an Extension is used as live
# ``Annotated[...]`` metadata on operation parameters; keep these modules PEP 563-free
# so the resolver sees the handle object rather than a string.

from types import SimpleNamespace
from typing import Annotated, Any, Callable

import pytest

from neofoam.framework.context import Context
from neofoam.framework.dependency_resolver import DependencyResolver
from neofoam.framework.model import (
    BoundExtension,
    Extension,
    Model,
    fold,
    negated,
)
from neofoam.framework.model.runtime import ModelRuntime
from neofoam.io import BaseConfig


class _ZoneConfig(BaseConfig):
    label: str


class _Sum:
    """Stands in for pybFoam's matrix types: traces ``+`` and ``-`` so the fold
    order is assertable, and raises ``TypeError`` for a foreign right operand
    instead of returning ``NotImplemented`` (the pybFoam operator behavior)."""

    def __init__(self, trace: str) -> None:
        self.trace = trace

    def _traced(self, other: Any) -> str:
        if not isinstance(other, _Sum):
            raise TypeError("incompatible function arguments")
        return other.trace

    def __add__(self, other: Any) -> "_Sum":
        return _Sum(f"({self.trace}+{self._traced(other)})")

    def __sub__(self, other: Any) -> "_Sum":
        return _Sum(f"({self.trace}-{self._traced(other)})")


def active_ctx(*runtimes: ModelRuntime) -> Context:
    """A Context in which exactly *runtimes* are active, keyed by their spec name."""
    return Context(fields={}, models={rt.spec.name: rt for rt in runtimes})


@pytest.fixture
def momentum_ext() -> Any:
    """An extension with one fold hook (its body combines the results onto a
    seed) and one broadcast hook, as an operation module would define it."""
    ext = Extension("momentum")

    @ext.defines
    def terms(U: str, contributions: list) -> Any:
        return fold(_Sum(f"zero({U})"), contributions)

    @ext.defines
    def constrain(UEqn: Any) -> None: ...

    return ext


def contributor(hook: Any, func: Callable[..., Any], name: str, config: Any = None) -> ModelRuntime:
    """A fresh ``Model`` named *name* contributing *func* to *hook*, returned as
    the live ``ModelRuntime`` that makes the contribution active for a case."""
    model = Model(name)
    model.contributes(hook)(func)
    return ModelRuntime(spec=model, name=name, config=config)


# ---------------------------------------------------------------------------
# Declaring hooks and contributing to them
# ---------------------------------------------------------------------------


def test_defines_returns_the_hook_handle_under_the_declared_name() -> None:
    ext = Extension("momentum")

    @ext.defines
    def terms(U: str) -> Any: ...

    assert terms.name == "terms"
    assert terms.extension is ext
    assert ext.hooks == {"terms": terms}


def test_hooks_are_reachable_as_extension_attributes(momentum_ext: Any) -> None:
    # Handle access for @contributes targets: two extensions can share a hook
    # name without colliding at module level.
    assert momentum_ext.terms is momentum_ext.hooks["terms"]
    with pytest.raises(AttributeError, match="extension 'momentum' defines no hook 'typo'"):
        momentum_ext.typo


def test_defining_the_same_hook_twice_raises() -> None:
    ext = Extension("momentum")

    def declare_terms() -> None:
        @ext.defines
        def terms(U: str) -> Any: ...

    declare_terms()
    with pytest.raises(RuntimeError, match="hook 'terms' is already defined"):
        declare_terms()


def test_contributes_returns_the_function_unchanged(momentum_ext: Any) -> None:
    hook = momentum_ext.hooks["terms"]
    mrf = Model("mrf")

    def mrf_terms(U: str) -> Any:
        return _Sum("mrf")

    assert mrf.contributes(hook)(mrf_terms) is mrf_terms


def test_contributes_rejects_a_bare_extension() -> None:
    mrf = Model("mrf")
    with pytest.raises(TypeError, match="must be a hook"):

        @mrf.contributes(Extension("momentum"))  # type: ignore[arg-type]
        def mrf_terms(U: str) -> Any:
            return _Sum("mrf")


# ---------------------------------------------------------------------------
# Hook dispatch: the declaration body combines the contribution results
# ---------------------------------------------------------------------------


def test_a_fold_hook_combines_the_seed_and_contributions_in_registration_order(
    momentum_ext: Any,
) -> None:
    hook = momentum_ext.hooks["terms"]
    mrf_rt = contributor(hook, lambda U: _Sum("mrf"), "mrf")
    fv_rt = contributor(hook, lambda U: _Sum("fvOptions"), "fvOptions")
    # Context insertion order deliberately reversed: registration order wins.
    ext = momentum_ext.resolve(active_ctx(fv_rt, mrf_rt))
    assert ext.terms("U").trace == "((zero(U)+mrf)+fvOptions)"


def test_a_negated_contribution_folds_by_subtraction(momentum_ext: Any) -> None:
    hook = momentum_ext.hooks["terms"]
    rt = contributor(hook, lambda U: negated(_Sum("src")), "fvOptions")
    ext = momentum_ext.resolve(active_ctx(rt))
    assert ext.terms("U").trace == "(zero(U)-src)"


def test_a_none_contribution_result_is_skipped_by_fold(momentum_ext: Any) -> None:
    hook = momentum_ext.hooks["terms"]
    quiet_rt = contributor(hook, lambda U: None, "quiet")
    mrf_rt = contributor(hook, lambda U: _Sum("mrf"), "mrf")
    ext = momentum_ext.resolve(active_ctx(quiet_rt, mrf_rt))
    assert ext.terms("U").trace == "(zero(U)+mrf)"


def test_a_fold_hook_without_active_contributions_returns_the_seed(momentum_ext: Any) -> None:
    ext = momentum_ext.resolve(Context(fields={}, models={}))
    assert ext.terms("U").trace == "zero(U)"


def test_an_inactive_model_does_not_contribute(momentum_ext: Any) -> None:
    hook = momentum_ext.hooks["terms"]
    contributor(hook, lambda U: _Sum("mrf"), "mrf")  # never put into the Context
    ext = momentum_ext.resolve(Context(fields={}, models={}))
    assert ext.terms("U").trace == "zero(U)"


def test_a_name_collision_with_a_non_runtime_does_not_activate(momentum_ext: Any) -> None:
    # Something else occupies the model's name in ctx.models: not a ModelRuntime,
    # so the model is not active and its contribution must not run.
    contributor(momentum_ext.hooks["terms"], lambda U: _Sum("mrf"), "mrf")
    ctx = Context(fields={}, models={"mrf": object()})
    assert momentum_ext.resolve(ctx).terms("U").trace == "zero(U)"


def test_a_runtime_of_a_different_spec_with_the_same_name_does_not_activate(
    momentum_ext: Any,
) -> None:
    # Activation is by ModelSpec identity, not by name: a same-named runtime of
    # another spec does not activate this contribution.
    contributor(momentum_ext.hooks["terms"], lambda U: _Sum("mrf"), "mrf")
    other_rt = ModelRuntime(spec=Model("mrf"), name="mrf", config=None)
    assert momentum_ext.resolve(active_ctx(other_rt)).terms("U").trace == "zero(U)"


def test_the_declaration_body_owns_the_combine_rule() -> None:
    # The same mechanism as a model interface's fold: the body receives every
    # result and applies its own rule — min with a declared empty-case default.
    ext = Extension("loop")

    @ext.defines
    def max_time_step(ceilings: list) -> float:
        return min(ceilings, default=1e300)

    slow_rt = contributor(ext.max_time_step, lambda: 0.5, "slow")
    fast_rt = contributor(ext.max_time_step, lambda: 0.2, "fast")
    assert ext.resolve(active_ctx(slow_rt, fast_rt)).max_time_step() == 0.2
    assert ext.resolve(Context(fields={}, models={})).max_time_step() == 1e300


def test_a_broadcast_hook_returns_the_raw_results_in_registration_order(
    momentum_ext: Any,
) -> None:
    hook = momentum_ext.hooks["constrain"]
    no_rt = contributor(hook, lambda UEqn: False, "passive")
    yes_rt = contributor(hook, lambda UEqn: True, "active")
    ext = momentum_ext.resolve(active_ctx(no_rt, yes_rt))
    assert ext.constrain("UEqn") == [False, True]
    assert any(ext.constrain("UEqn"))


def test_a_broadcast_hook_never_invokes_the_declaration_body() -> None:
    ext = Extension("momentum")

    @ext.defines
    def constrain(UEqn: Any) -> None:
        raise AssertionError("a broadcast hook body must not run")

    assert ext.resolve(Context(fields={}, models={})).constrain("UEqn") == []


def test_a_call_missing_a_hook_argument_raises_naming_the_hook(momentum_ext: Any) -> None:
    ext = momentum_ext.resolve(Context(fields={}, models={}))
    with pytest.raises(TypeError, match=r"hook 'momentum.terms' called without argument\(s\): U"):
        ext.terms()


# ---------------------------------------------------------------------------
# Contribution parameter resolution
# ---------------------------------------------------------------------------


def test_a_contribution_receives_the_call_argument_by_name(momentum_ext: Any) -> None:
    hook = momentum_ext.hooks["terms"]
    rt = contributor(hook, lambda U: _Sum(f"ddt({U})"), "mrf")
    ext = momentum_ext.resolve(active_ctx(rt))
    assert ext.terms("field-U").trace == "(zero(field-U)+ddt(field-U))"


def _hook_from_zones(U: str, mrf_zones: Annotated[Any, "models"]) -> Any:
    return _Sum(mrf_zones.name)


def _hook_from_config(U: str, cfg: _ZoneConfig) -> Any:
    return _Sum(cfg.label)


def test_a_contribution_resolves_a_models_annotated_parameter(momentum_ext: Any) -> None:
    hook = momentum_ext.hooks["terms"]
    rt = contributor(hook, _hook_from_zones, "mrf")
    zones = SimpleNamespace(name="zones-from-context")
    ctx = Context(fields={}, models={"mrf": rt, "mrf_zones": zones})
    assert momentum_ext.resolve(ctx).terms("U").trace == "(zero(U)+zones-from-context)"


def test_a_contribution_resolves_a_config_parameter_from_its_own_runtime(
    momentum_ext: Any,
) -> None:
    hook = momentum_ext.hooks["terms"]
    rt = contributor(hook, _hook_from_config, "mrf", config=_ZoneConfig(label="rotor"))
    assert momentum_ext.resolve(active_ctx(rt)).terms("U").trace == "(zero(U)+rotor)"


def _hook_needs_field(U: str, deltaT: float) -> Any:
    return _Sum(str(deltaT))


def test_a_missing_contribution_parameter_error_names_hook_and_contribution(
    momentum_ext: Any,
) -> None:
    hook = momentum_ext.hooks["terms"]
    rt = contributor(hook, _hook_needs_field, "mrf")
    with pytest.raises(ValueError) as excinfo:
        momentum_ext.resolve(active_ctx(rt)).terms("U")
    message = str(excinfo.value)
    assert "momentum.terms" in message
    assert "_hook_needs_field" in message
    assert "deltaT" in message


# ---------------------------------------------------------------------------
# The bound handle and resolver injection
# ---------------------------------------------------------------------------


def test_an_unknown_hook_raises_naming_the_extension(momentum_ext: Any) -> None:
    ext = momentum_ext.resolve(Context(fields={}, models={}))
    with pytest.raises(AttributeError, match="extension 'momentum' defines no hook 'typo'"):
        ext.typo()


def test_underscore_attributes_never_dispatch_on_a_bound_extension(momentum_ext: Any) -> None:
    ext = momentum_ext.resolve(Context(fields={}, models={}))
    with pytest.raises(AttributeError):
        ext._not_a_hook


def test_resolver_injects_the_bound_extension_for_an_annotated_param(momentum_ext: Any) -> None:
    hook = momentum_ext.hooks["terms"]
    rt = contributor(hook, lambda U: _Sum("mrf"), "mrf")

    def momentum(self: Any, ext: Annotated[BoundExtension, momentum_ext]) -> str:
        return str(ext.terms("U").trace)

    kwargs = DependencyResolver().resolve_arguments(momentum, ctx=active_ctx(rt))
    assert isinstance(kwargs["ext"], BoundExtension)
    assert momentum(None, kwargs["ext"]) == "(zero(U)+mrf)"


def test_resolver_raises_for_an_extension_param_without_a_context(momentum_ext: Any) -> None:
    def momentum(self: Any, ext: Annotated[BoundExtension, momentum_ext]) -> Any:
        return ext

    with pytest.raises(ValueError, match="no Context"):
        DependencyResolver().resolve_arguments(momentum, ctx=None)
