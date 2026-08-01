# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for operation-declared extension points: declaring a point, registering
implementation factories on it via ``@<model>.extends``, and resolving the active
implementations for one Context into the injected ``Extensions`` container."""

# NOTE: no `from __future__ import annotations` — an ExtensionPoint is used as live
# ``Annotated[...]`` metadata on operation parameters; keep these modules PEP 563-free
# so the resolver sees the handle object rather than a string.

from types import SimpleNamespace
from typing import Annotated, Any, Callable

import pytest

from neofoam.framework.context import Context
from neofoam.framework.dependency_resolver import DependencyResolver
from neofoam.framework.model import (
    ExtensionPoint,
    Extensions,
    Model,
    negated,
)
from neofoam.framework.model.runtime import ModelRuntime
from neofoam.io import BaseConfig


class _ZoneConfig(BaseConfig):
    label: str


class _Correction:
    """The interface an operation module declares: one method per call site."""

    def label(self) -> str:
        return "none"

    def terms(self) -> list[str]:
        return []


class _NamedCorrection(_Correction):
    def __init__(self, label: str) -> None:
        self._label = label

    def label(self) -> str:
        return self._label


@pytest.fixture
def point() -> Any:
    """A fresh extension point as an operation module would declare it."""
    return ExtensionPoint("velocity_correction", _Correction)


def implementor(
    target: Any, factory: Callable[..., Any], name: str, config: Any = None
) -> ModelRuntime:
    """A fresh ``Model`` named *name* registering *factory* on *target*, returned as
    the live ``ModelRuntime`` that makes the factory active for a case."""
    model = Model(name)
    model.extends(target)(factory)
    return ModelRuntime(spec=model, name=name, config=config)


def active_ctx(*runtimes: ModelRuntime) -> Context:
    """A Context in which exactly *runtimes* are active, keyed by their spec name."""
    return Context(fields={}, models={rt.spec.name: rt for rt in runtimes})


# Named factories reused by the kwargs / error-message families: their `__name__`
# and parameter annotations are load-bearing (unlike the anonymous lambdas used
# where only the produced implementation matters).
def _from_zones(mrf_zones: Annotated[Any, "models"]) -> _Correction:
    return _NamedCorrection(mrf_zones.name)


def _from_config(cfg: _ZoneConfig) -> _Correction:
    return _NamedCorrection(cfg.label)


def _needs_field(deltaT: float) -> _Correction:
    return _NamedCorrection(str(deltaT))


# ---------------------------------------------------------------------------
# Declaring an extension point
# ---------------------------------------------------------------------------


def test_extension_point_carries_its_name_and_interface(point: Any) -> None:
    assert point.name == "velocity_correction"
    assert point.protocol is _Correction


def test_a_fresh_extension_point_has_no_factories(point: Any) -> None:
    assert point.factories == ()


# ---------------------------------------------------------------------------
# Registering implementation factories
# ---------------------------------------------------------------------------


def test_extends_registers_the_factory_against_the_point(point: Any) -> None:
    mrf = Model("mrf")

    @mrf.extends(point)
    def make_mrf_correction() -> _Correction:
        return _NamedCorrection("mrf")

    assert make_mrf_correction in point.factories


def test_extends_records_the_registering_model_as_owner(point: Any) -> None:
    mrf = Model("mrf")

    @mrf.extends(point)
    def make_mrf_correction() -> _Correction:
        return _NamedCorrection("mrf")

    assert point.owner_of(make_mrf_correction) is mrf


def test_extends_returns_the_function_unchanged(point: Any) -> None:
    mrf = Model("mrf")

    def make_mrf_correction() -> _Correction:
        return _NamedCorrection("mrf")

    returned = mrf.extends(point)(make_mrf_correction)
    assert returned is make_mrf_correction


def test_factories_are_returned_in_registration_order(point: Any) -> None:
    mrf = Model("mrf")
    fv_options = Model("fvOptions")

    @mrf.extends(point)
    def make_mrf_correction() -> _Correction:
        return _NamedCorrection("mrf")

    @fv_options.extends(point)
    def make_fv_options_correction() -> _Correction:
        return _NamedCorrection("fvOptions")

    assert point.factories == (make_mrf_correction, make_fv_options_correction)


def test_extends_against_a_non_extension_point_target_raises() -> None:
    mrf = Model("mrf")
    with pytest.raises(TypeError, match="must be an ExtensionPoint"):

        @mrf.extends(_Correction)  # type: ignore[arg-type]
        def make_mrf_correction() -> _Correction:
            return _NamedCorrection("mrf")


def test_owner_of_unregistered_factory_raises(point: Any) -> None:
    def never_registered() -> _Correction:
        return _NamedCorrection("nope")

    with pytest.raises(KeyError, match="not a registered factory"):
        point.owner_of(never_registered)


# ---------------------------------------------------------------------------
# Resolving the active implementations for one Context
# ---------------------------------------------------------------------------


def test_resolve_builds_the_implementation_of_an_active_model(point: Any) -> None:
    rt = implementor(point, lambda: _NamedCorrection("mrf"), "mrf")
    extensions = point.resolve(active_ctx(rt))
    assert [e.label() for e in extensions] == ["mrf"]


def test_resolve_skips_a_model_absent_from_the_context(point: Any) -> None:
    implementor(point, lambda: _NamedCorrection("mrf"), "mrf")
    assert list(point.resolve(Context(fields={}, models={}))) == []


def test_resolve_skips_a_name_collision_with_a_non_runtime(point: Any) -> None:
    # Something else occupies the model's name in ctx.models: not a ModelRuntime,
    # so the model is not active and its factory must not run.
    implementor(point, lambda: _NamedCorrection("mrf"), "mrf")
    ctx = Context(fields={}, models={"mrf": object()})
    assert list(point.resolve(ctx)) == []


def test_resolve_skips_a_runtime_of_a_different_spec_with_the_same_name(point: Any) -> None:
    # Activation is by ModelSpec identity, not by name: a same-named runtime of
    # another spec does not activate this factory.
    implementor(point, lambda: _NamedCorrection("mrf"), "mrf")
    other_rt = ModelRuntime(spec=Model("mrf"), name="mrf", config=None)
    assert list(point.resolve(active_ctx(other_rt))) == []


def test_resolve_returns_implementations_in_registration_order(point: Any) -> None:
    mrf_rt = implementor(point, lambda: _NamedCorrection("mrf"), "mrf")
    fv_rt = implementor(point, lambda: _NamedCorrection("fvOptions"), "fvOptions")
    # Context insertion order deliberately reversed: registration order wins.
    extensions = point.resolve(active_ctx(fv_rt, mrf_rt))
    assert [e.label() for e in extensions] == ["mrf", "fvOptions"]


def test_resolve_includes_only_the_active_subset(point: Any) -> None:
    mrf_rt = implementor(point, lambda: _NamedCorrection("mrf"), "mrf")
    implementor(point, lambda: _NamedCorrection("fvOptions"), "fvOptions")
    extensions = point.resolve(active_ctx(mrf_rt))
    assert [e.label() for e in extensions] == ["mrf"]


def test_resolve_rebuilds_the_implementations_on_every_injection(point: Any) -> None:
    # Implementations are cheap wrappers rebuilt per injection — nothing live is
    # cached on the module-level point.
    rt = implementor(point, lambda: _NamedCorrection("mrf"), "mrf")
    ctx = active_ctx(rt)
    first = list(point.resolve(ctx))
    second = list(point.resolve(ctx))
    assert first[0] is not second[0]


# ---------------------------------------------------------------------------
# The injected container
# ---------------------------------------------------------------------------


def test_empty_extensions_is_falsy_and_empty(point: Any) -> None:
    extensions = point.resolve(Context(fields={}, models={}))
    assert isinstance(extensions, Extensions)
    assert not extensions
    assert len(extensions) == 0


def test_non_empty_extensions_is_truthy_and_counted(point: Any) -> None:
    mrf_rt = implementor(point, lambda: _NamedCorrection("mrf"), "mrf")
    fv_rt = implementor(point, lambda: _NamedCorrection("fvOptions"), "fvOptions")
    extensions = point.resolve(active_ctx(mrf_rt, fv_rt))
    assert extensions
    assert len(extensions) == 2


def test_extensions_can_be_iterated_more_than_once(point: Any) -> None:
    rt = implementor(point, lambda: _NamedCorrection("mrf"), "mrf")
    extensions = point.resolve(active_ctx(rt))
    assert [e.label() for e in extensions] == ["mrf"]
    assert [e.label() for e in extensions] == ["mrf"]


# ---------------------------------------------------------------------------
# Composite call sites: one call per site, terms fold into an expression
# ---------------------------------------------------------------------------


class _Sum:
    """Stands in for pybFoam's matrix types: raises ``TypeError`` for a foreign
    right operand instead of returning ``NotImplemented`` — the behavior the
    fold's subclass-of-the-sum-type dispatch exists for."""

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


class _Momentum:
    """An interface with one void site and one term site."""

    def record(self, log: list[str]) -> None: ...

    def terms(self) -> list[Any]:
        return []


class _Contributor(_Momentum):
    def __init__(self, label: str) -> None:
        self._label = label

    def record(self, log: list[str]) -> None:
        log.append(self._label)

    def terms(self) -> list[Any]:
        return [_Sum(self._label)]


class _RhsContributor(_Momentum):
    """Contributes its term as a source: native's ``==``, folded with ``-``."""

    def terms(self) -> list[Any]:
        return [negated(_Sum("src"))]


@pytest.fixture
def fold_point() -> Any:
    """A point whose term sites fold into ``_Sum`` expressions."""
    return ExtensionPoint("momentum_site", _Momentum, folds_into=_Sum)


def test_a_site_call_fans_out_in_registration_order(fold_point: Any) -> None:
    mrf_rt = implementor(fold_point, lambda: _Contributor("mrf"), "mrf")
    fv_rt = implementor(fold_point, lambda: _Contributor("fvOptions"), "fvOptions")
    log: list[str] = []
    fold_point.resolve(active_ctx(fv_rt, mrf_rt)).record(log)
    assert log == ["mrf", "fvOptions"]


def test_a_term_site_folds_into_the_sum_in_registration_order(fold_point: Any) -> None:
    mrf_rt = implementor(fold_point, lambda: _Contributor("mrf"), "mrf")
    fv_rt = implementor(fold_point, lambda: _Contributor("fvOptions"), "fvOptions")
    ext = fold_point.resolve(active_ctx(mrf_rt, fv_rt))
    assert (_Sum("seed") + ext.terms()).trace == "((seed+mrf)+fvOptions)"


def test_a_negated_term_folds_by_subtraction(fold_point: Any) -> None:
    rt = implementor(fold_point, lambda: _RhsContributor(), "fvOptions")
    ext = fold_point.resolve(active_ctx(rt))
    assert (_Sum("seed") + ext.terms()).trace == "(seed-src)"


def test_an_empty_fold_leaves_the_sum_untouched(fold_point: Any) -> None:
    # Identity, not a neutral element: no arithmetic happens at all, so an
    # inactive point can never perturb the sum.
    ext = fold_point.resolve(Context(fields={}, models={}))
    seed = _Sum("seed")
    assert (seed + ext.terms()) is seed


def test_a_site_call_without_folds_into_returns_none(point: Any) -> None:
    rt = implementor(point, lambda: _NamedCorrection("mrf"), "mrf")
    assert point.resolve(active_ctx(rt)).terms() is None


def test_an_unknown_site_raises_naming_the_interface(fold_point: Any) -> None:
    ext = fold_point.resolve(Context(fields={}, models={}))
    with pytest.raises(AttributeError, match="'_Momentum' declares no extension site 'typo'"):
        ext.typo()


def test_underscore_attributes_never_dispatch(fold_point: Any) -> None:
    ext = fold_point.resolve(Context(fields={}, models={}))
    with pytest.raises(AttributeError):
        ext._not_a_site


# ---------------------------------------------------------------------------
# Factory parameter resolution
# ---------------------------------------------------------------------------


def test_factory_resolves_a_models_annotated_parameter(point: Any) -> None:
    rt = implementor(point, _from_zones, "mrf")
    zones = SimpleNamespace(name="zones-from-context")
    ctx = Context(fields={}, models={"mrf": rt, "mrf_zones": zones})
    assert [e.label() for e in point.resolve(ctx)] == ["zones-from-context"]


def test_factory_resolves_a_config_parameter_from_its_own_runtime(point: Any) -> None:
    rt = implementor(point, _from_config, "mrf", config=_ZoneConfig(label="rotor"))
    assert [e.label() for e in point.resolve(active_ctx(rt))] == ["rotor"]


@pytest.mark.parametrize(
    "factory, config, missing_param",
    [
        (_needs_field, None, "deltaT"),  # field param absent from the Context
        (_from_config, None, "cfg"),  # config param but the runtime has no config
    ],
)
def test_missing_factory_parameter_error_names_point_factory_param(
    point: Any, factory: Callable[..., Any], config: Any, missing_param: str
) -> None:
    rt = implementor(point, factory, "mrf", config=config)
    with pytest.raises(ValueError) as excinfo:
        point.resolve(active_ctx(rt))
    message = str(excinfo.value)
    assert point.name in message
    assert factory.__name__ in message
    assert missing_param in message


# ---------------------------------------------------------------------------
# Resolver injection into an operation
# ---------------------------------------------------------------------------


def test_resolver_injects_the_active_extensions_for_an_annotated_param(point: Any) -> None:
    rt = implementor(point, lambda: _NamedCorrection("mrf"), "mrf")

    def momentum(self: Any, ext: Annotated[Extensions[_Correction], point]) -> list[str]:
        return [e.label() for e in ext]

    kwargs = DependencyResolver().resolve_arguments(momentum, ctx=active_ctx(rt))
    assert isinstance(kwargs["ext"], Extensions)
    assert momentum(None, kwargs["ext"]) == ["mrf"]


def test_resolver_injects_an_empty_container_when_no_model_is_active(point: Any) -> None:
    implementor(point, lambda: _NamedCorrection("mrf"), "mrf")

    def momentum(self: Any, ext: Annotated[Extensions[_Correction], point]) -> list[str]:
        return [e.label() for e in ext]

    kwargs = DependencyResolver().resolve_arguments(momentum, ctx=Context(fields={}, models={}))
    assert not kwargs["ext"]


def test_resolver_raises_for_an_extension_point_param_without_a_context(point: Any) -> None:
    def momentum(self: Any, ext: Annotated[Extensions[_Correction], point]) -> list[str]:
        return [e.label() for e in ext]

    with pytest.raises(ValueError, match="no Context"):
        DependencyResolver().resolve_arguments(momentum, ctx=None)
