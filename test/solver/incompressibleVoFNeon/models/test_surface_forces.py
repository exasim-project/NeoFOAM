# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Unit tests for the surfaceForces interface model (pure Python, no mesh).

The fold semantics and the default-contributor wiring are the bitwise-parity
guarantees of the NeoN VoF solver: surface tension is registered before
gravity so the folded sum reproduces the legacy inline
``fSigma + (-1.0*ghf)*snGrad(rho)`` addition order.
"""

from types import SimpleNamespace
from typing import Any

import pytest

import neofoam.solver.incompressibleVoFNeon.models.surface_forces as sf
from neofoam.framework.context import Context
from neofoam.framework.model import Model, ModelRuntime
from neofoam.framework.model.interface import bind_owned_interfaces


def test_interface_is_owned_by_surface_forces_model() -> None:
    assert sf.interfaceForce.owner is sf.surfaceForces
    assert sf.interfaceForce.name == "interfaceForce"


def test_fold_sums_in_registration_order() -> None:
    # Strings: non-commutative "+" pins left-to-right order, not just the sum.
    assert sf.interfaceForce.fold(["st", "g"]) == "stg"
    assert sf.interfaceForce.fold([3.0, -10.0]) == pytest.approx(-7.0)


def test_fold_empty_returns_none() -> None:
    assert sf.interfaceForce.fold([]) is None


def test_default_contribution_order_is_surface_tension_then_gravity() -> None:
    # Prefix assertion: other tests may register extra contributions on the
    # module-level interface; the two defaults must stay first and ordered.
    owners = [
        sf.interfaceForce.owner_of(f).name for f in sf.interfaceForce.contributions
    ]
    assert owners[:2] == ["surfaceTensionForce", "gravityForce"]


def _live_context() -> Context:
    return Context(
        fields={"alpha1": object(), "rho": object(), "ghf": 2.0},
        models={"phase": {"sigma": 0.07}, "neon_runtime": object()},
    )


def _default_runtimes() -> list[ModelRuntime]:
    return [
        ModelRuntime(
            spec=sf.surfaceTensionForce, name="surfaceTensionForce", config=None
        ),
        ModelRuntime(spec=sf.gravityForce, name="gravityForce", config=None),
    ]


def test_bound_fold_matches_inline_expression(monkeypatch: pytest.MonkeyPatch) -> None:
    """Folded defaults == fSigma + (-1.0*ghf)*snGrad(rho) with scalar stand-ins."""
    monkeypatch.setattr(
        sf,
        "nfb",
        SimpleNamespace(
            surface_tension_force=lambda rt, alpha1, sigma: 3.0,
            sn_grad=lambda rho: 5.0,
        ),
    )
    owner_rt = ModelRuntime(spec=sf.surfaceForces, name="surfaceForces", config=None)
    bind_owned_interfaces(owner_rt, _default_runtimes(), Context(fields={}, models={}))

    folded = owner_rt.bound_interfaces["interfaceForce"](_live_context())
    assert folded == pytest.approx(3.0 + (-1.0 * 2.0) * 5.0)


def test_third_force_model_folds_in_without_owner_edit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A new force model contributes without any edit to owner or consumers."""
    monkeypatch.setattr(
        sf,
        "nfb",
        SimpleNamespace(
            surface_tension_force=lambda rt, alpha1, sigma: 3.0,
            sn_grad=lambda rho: 5.0,
        ),
    )
    dummy_force = Model("dummyMarangoniForce")

    @dummy_force.contributes(sf.interfaceForce)
    def marangoni() -> Any:
        return 0.25

    owner_with = ModelRuntime(spec=sf.surfaceForces, name="surfaceForces", config=None)
    candidates = _default_runtimes() + [
        ModelRuntime(spec=dummy_force, name="dummyMarangoniForce", config=None)
    ]
    bind_owned_interfaces(owner_with, candidates, Context(fields={}, models={}))
    assert owner_with.bound_interfaces["interfaceForce"](
        _live_context()
    ) == pytest.approx(-7.0 + 0.25)

    # Without the dummy runtime the contribution is inactive and excluded.
    owner_without = ModelRuntime(
        spec=sf.surfaceForces, name="surfaceForces", config=None
    )
    bind_owned_interfaces(
        owner_without, _default_runtimes(), Context(fields={}, models={})
    )
    assert owner_without.bound_interfaces["interfaceForce"](
        _live_context()
    ) == pytest.approx(-7.0)
