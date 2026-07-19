# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for turbulence model selection under the merged family + ``fallback`` flag.

One family serves both solvers; the ``fallback`` flag picks the backend:

* ``fallback=False`` (incompressibleFluidNeoN) → a native
  :class:`~neofoam.turbulence.native.NeoNHandle` for a registered model.
* ``fallback=True`` (incompressibleFluid) → a
  :class:`~neofoam.turbulence.fallback.FallbackHandle` for a model with a
  co-located ``fallback=True`` op.

Parametrized over the discovered cases: a ``native`` case is a registered dual
model (both branches build); an ``unregistered`` case (e.g. Smagorinsky) has no
spec, so **both** branches raise. This also exercises the pybFoam-bound
``select_from_case`` entry point.
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

from turbulence.conftest import CASES, Case


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_native_branch(case: Case) -> None:
    cfg = TurbulencePropertiesConfig.load(case_dir=case.path)
    if case.selection["resolves_to"] == "native":
        selected = select_turbulence_model(cfg, fallback=False, case_dir=case.path)
        assert isinstance(selected, NeoNHandle)
    else:
        with pytest.raises(ValueError):
            select_turbulence_model(cfg, fallback=False, case_dir=case.path)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_fallback_branch(case: Case) -> None:
    cfg = TurbulencePropertiesConfig.load(case_dir=case.path)
    if case.selection["resolves_to"] == "native":  # dual model → has a fallback op
        selected = select_turbulence_model(
            cfg, fallback=True, case_dir=case.path, of_factory=MagicMock()
        )
        assert isinstance(selected, FallbackHandle)
    else:  # unregistered → no spec → raises
        with pytest.raises(ValueError):
            select_turbulence_model(
                cfg, fallback=True, case_dir=case.path, of_factory=MagicMock()
            )


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_select_from_case_fallback(case: Case) -> None:
    if case.selection["resolves_to"] == "native":
        selected = select_from_case(case.path, fallback=True, of_factory=MagicMock())
        assert isinstance(selected, FallbackHandle)
    else:
        with pytest.raises(ValueError):
            select_from_case(case.path, fallback=True, of_factory=MagicMock())


def test_unregistered_model_raises_clearly() -> None:
    cfg = SimpleNamespace(
        simulationType="LES", RAS=None, LES=SimpleNamespace(LESModel="Smagorinsky")
    )
    with pytest.raises(ValueError, match="no turbulence model registered"):
        select_turbulence_model(cfg, fallback=True)


def test_model_name_undeterminable_is_none() -> None:
    # The positive paths (laminar / RASModel / LESModel resolution) are covered,
    # manifest driven, by test_config.py; here we only pin the unknown case.
    assert model_name(SimpleNamespace(simulationType="bogus")) is None
