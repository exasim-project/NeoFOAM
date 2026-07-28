# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for turbulence model selection under the merged family + ``fallback`` flag.

One family serves both solvers; the ``fallback`` flag picks the backend:

* ``fallback=False`` (incompressibleFluidNeoN) → a native
  :class:`~neofoam.turbulence.native.NeoNHandle` for a registered model.
* ``fallback=True`` (incompressibleFluid) → a
  :class:`~neofoam.turbulence.fallback.FallbackHandle`, built either from the
  model's co-located ``fallback=True`` op (registered) or straight from
  OpenFOAM's own run-time selection table (a name with no registered spec).

Parametrized over the discovered cases, whose ``expected.yaml`` declares how each
resolves: ``native`` is a registered dual model (both backends build);
``unregistered`` (Smagorinsky) has no spec, so only the pybFoam backend builds it;
``wrong_family`` (``LES { LESModel kEpsilon; }``) names a model registered in the
*other* family, which must be refused on both backends rather than silently built
as the RAS closure. This also exercises the pybFoam-bound ``select_from_case``
entry point.

The pybFoam turbulence factory is injected as a ``MagicMock`` throughout: what is
under test is which model selection picks, not what OpenFOAM then does with it —
so no mesh is needed.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from neofoam.turbulence.config import TurbulencePropertiesConfig
from neofoam.turbulence.fallback import FallbackHandle
from neofoam.turbulence.native import NeoNHandle
from neofoam.turbulence.selection import (
    model_name,
    select_from_case,
    select_turbulence_model,
)
from turbulence.conftest import CASES, Case, case_for

#: One case per resolution kind, taken from the discovered manifests.
REGISTERED = case_for("kEpsilon")
UNREGISTERED = next(c for c in CASES if c.selection["resolves_to"] == "unregistered")
WRONG_FAMILY = next(c for c in CASES if c.selection["resolves_to"] == "wrong_family")


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_native_branch(case: Case) -> None:
    cfg = TurbulencePropertiesConfig.load(case_dir=case.path)
    if case.selection["resolves_to"] == "native":
        selected = select_turbulence_model(cfg, fallback=False, case_dir=case.path)
        assert isinstance(selected, NeoNHandle)
    else:  # unregistered → no native closure; wrong_family → the family guard
        with pytest.raises(ValueError):
            select_turbulence_model(cfg, fallback=False, case_dir=case.path)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_fallback_branch(case: Case) -> None:
    cfg = TurbulencePropertiesConfig.load(case_dir=case.path)
    if case.selection["resolves_to"] == "wrong_family":
        with pytest.raises(ValueError):
            select_turbulence_model(cfg, fallback=True, case_dir=case.path, of_factory=MagicMock())
    else:  # a registered dual model, or an unregistered name OpenFOAM can build
        selected = select_turbulence_model(
            cfg, fallback=True, case_dir=case.path, of_factory=MagicMock()
        )
        assert isinstance(selected, FallbackHandle)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_select_from_case_fallback(case: Case) -> None:
    if case.selection["resolves_to"] == "wrong_family":
        with pytest.raises(ValueError):
            select_from_case(case.path, fallback=True, of_factory=MagicMock())
    else:
        selected = select_from_case(case.path, fallback=True, of_factory=MagicMock())
        assert isinstance(selected, FallbackHandle)


def test_registered_model_keeps_its_own_fallback_correct_op() -> None:
    cfg = TurbulencePropertiesConfig.load(case_dir=REGISTERED.path)

    handle = select_turbulence_model(
        cfg, fallback=True, case_dir=REGISTERED.path, of_factory=MagicMock()
    )

    assert [op.metadata.op_name for op in handle.operations] == ["kEpsilonCorrect"]


def test_unregistered_model_gets_the_openfoam_correct_op() -> None:
    cfg = TurbulencePropertiesConfig.load(case_dir=UNREGISTERED.path)

    handle = select_turbulence_model(
        cfg, fallback=True, case_dir=UNREGISTERED.path, of_factory=MagicMock()
    )

    assert [op.metadata.op_name for op in handle.operations] == ["of_correct_turbulence"]


def test_unregistered_model_builds_through_the_openfoam_factory() -> None:
    cfg = TurbulencePropertiesConfig.load(case_dir=UNREGISTERED.path)
    factory = MagicMock()
    handle = select_turbulence_model(
        cfg, fallback=True, case_dir=UNREGISTERED.path, of_factory=factory
    )

    handle.build()

    factory.assert_called_once()


def test_unregistered_model_selection_is_announced(capsys: pytest.CaptureFixture[str]) -> None:
    cfg = TurbulencePropertiesConfig.load(case_dir=UNREGISTERED.path)

    select_turbulence_model(cfg, fallback=True, case_dir=UNREGISTERED.path, of_factory=MagicMock())

    announcement = capsys.readouterr().out
    assert UNREGISTERED.selection["model_name"] in announcement
    assert "fallback" in announcement


def test_unregistered_model_has_no_native_closure() -> None:
    cfg = TurbulencePropertiesConfig.load(case_dir=UNREGISTERED.path)

    with pytest.raises(ValueError, match="no turbulence model registered"):
        select_turbulence_model(cfg, fallback=False, case_dir=UNREGISTERED.path)


def test_wrong_family_selection_names_the_registered_family() -> None:
    cfg = TurbulencePropertiesConfig.load(case_dir=WRONG_FAMILY.path)

    with pytest.raises(ValueError, match="registered as a RAS closure"):
        select_turbulence_model(
            cfg, fallback=True, case_dir=WRONG_FAMILY.path, of_factory=MagicMock()
        )


def test_undeterminable_model_raises() -> None:
    # Not expressible on disk: TurbulencePropertiesConfig pins simulationType to
    # the closed set, so an unknown one only reaches the selector duck-typed.
    cfg = SimpleNamespace(simulationType="bogus")

    with pytest.raises(ValueError, match="cannot resolve the turbulence model"):
        select_turbulence_model(cfg, fallback=True, of_factory=MagicMock())


def test_model_name_undeterminable_is_none() -> None:
    # The positive paths (laminar / RASModel / LESModel resolution) are covered,
    # manifest driven, by test_config.py; here we only pin the unknown case.
    assert model_name(SimpleNamespace(simulationType="bogus")) is None
