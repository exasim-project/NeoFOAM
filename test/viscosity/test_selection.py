# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for viscosity model selection (native match vs OpenFOAM fallback).

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

from neofoam.viscosity.config import TransportPropertiesConfig
from neofoam.viscosity.selection import (
    model_name,
    select_from_case,
    select_viscosity_model,
)

from viscosity.conftest import CASES, Case, assert_selection


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_select_from_loaded_config(case: Case) -> None:
    cfg = TransportPropertiesConfig.load(case_dir=case.path)
    selected = select_viscosity_model(cfg, of_factory=MagicMock())
    assert_selection(selected, case)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_select_from_case(case: Case) -> None:
    selected = select_from_case(case.path, of_factory=MagicMock())
    assert_selection(selected, case)


def test_model_name_missing_transport_model_is_none() -> None:
    # The positive path (a real transportModel resolves) is covered, manifest
    # driven, by test_config.py; here we only pin the not-determinable case.
    assert model_name(SimpleNamespace()) is None
