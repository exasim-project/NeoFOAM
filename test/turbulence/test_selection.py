# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for turbulence model selection (native match vs OpenFOAM fallback).

Parametrized over the discovered cases: the config is loaded from each real
case, and the selected model's name/kind is checked against the case manifest.
A registered native being dispatched (not falling back) is covered by the real
``resolves_to: native`` cases — no throwaway registration. Each native model
added (with its case) extends this automatically.

This also exercises the pybFoam-bound ``select_from_case`` entry point (absorbed
from the old ``test_selection_from_case.py``).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from neofoam.turbulence.config import TurbulencePropertiesConfig
from neofoam.turbulence.selection import (
    model_name,
    select_from_case,
    select_turbulence_model,
)

from turbulence.conftest import CASES, Case, assert_selection


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_select_from_loaded_config(case: Case) -> None:
    cfg = TurbulencePropertiesConfig.load(case_dir=case.path)
    selected = select_turbulence_model(cfg, of_factory=MagicMock())
    assert_selection(selected, case)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_select_from_case(case: Case) -> None:
    selected = select_from_case(case.path, of_factory=MagicMock())
    assert_selection(selected, case)


def test_model_name_undeterminable_is_none() -> None:
    # The positive paths (laminar / RASModel / LESModel resolution) are covered,
    # manifest driven, by test_config.py; here we only pin the unknown case.
    assert model_name(SimpleNamespace(simulationType="bogus")) is None
