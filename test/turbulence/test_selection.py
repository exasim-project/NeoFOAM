# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for turbulence model selection (native match vs OpenFOAM fallback).

Configs are duck-typed ``SimpleNamespace`` objects, so these tests run without
``neofoam.io`` / pybFoam.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from neofoam.framework.model import ModelSpec
from neofoam.turbulence.fallback import OpenFOAMTurbulenceModel
from neofoam.turbulence.interface import Model, turbulenceModel
from neofoam.turbulence.selection import model_name, select_turbulence_model

# Register the bundled natives (laminar).
import neofoam.turbulence.models  # noqa: F401


def _ras(model: str) -> SimpleNamespace:
    return SimpleNamespace(
        simulationType="RAS",
        RAS=SimpleNamespace(RASModel=model),
        LES=None,
    )


def _les(model: str) -> SimpleNamespace:
    return SimpleNamespace(
        simulationType="LES",
        RAS=None,
        LES=SimpleNamespace(LESModel=model),
    )


def test_model_name_resolves_per_simulation_type() -> None:
    assert model_name(SimpleNamespace(simulationType="laminar")) == "laminar"
    assert model_name(_ras("kEpsilon")) == "kEpsilon"
    assert model_name(_les("Smagorinsky")) == "Smagorinsky"
    assert model_name(SimpleNamespace(simulationType="bogus")) is None


def test_select_returns_native_spec_for_laminar() -> None:
    selected = select_turbulence_model(SimpleNamespace(simulationType="laminar"))
    assert isinstance(selected, ModelSpec)
    assert selected.name == "laminar"


def test_select_returns_native_for_registered_ras_model(
    clean_turbulence_registry: None,
) -> None:
    k_epsilon = Model("kEpsilon").register_with(turbulenceModel)

    selected = select_turbulence_model(_ras("kEpsilon"))

    assert isinstance(selected, ModelSpec)
    assert selected is k_epsilon


def test_select_falls_back_to_openfoam_for_unknown_ras_model() -> None:
    factory = MagicMock()
    selected = select_turbulence_model(_ras("kOmegaSST"), of_factory=factory)
    assert isinstance(selected, OpenFOAMTurbulenceModel)


def test_select_falls_back_to_openfoam_for_unknown_les_model() -> None:
    selected = select_turbulence_model(_les("SpalartAllmarasDDES"))
    assert isinstance(selected, OpenFOAMTurbulenceModel)
