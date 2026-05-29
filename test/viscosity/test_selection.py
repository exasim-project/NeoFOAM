# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for viscosity model selection (native match vs OpenFOAM fallback).

Configs are duck-typed ``SimpleNamespace`` objects, so these tests run without
``neofoam.io`` / pybFoam.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from neofoam.framework.model import ModelSpec
from neofoam.viscosity.fallback import OpenFOAMViscosityModel
from neofoam.viscosity.interface import Model, viscosityModel
from neofoam.viscosity.selection import model_name, select_viscosity_model

# Register the bundled natives (Newtonian).
import neofoam.viscosity.models  # noqa: F401


def _cfg(transport_model: str) -> SimpleNamespace:
    return SimpleNamespace(transportModel=transport_model)


def test_model_name_resolves_transport_model() -> None:
    assert model_name(_cfg("Newtonian")) == "Newtonian"
    assert model_name(_cfg("CrossPowerLaw")) == "CrossPowerLaw"
    assert model_name(SimpleNamespace()) is None


def test_select_returns_native_spec_for_newtonian() -> None:
    selected = select_viscosity_model(_cfg("Newtonian"))
    assert isinstance(selected, ModelSpec)
    assert selected.name == "Newtonian"


def test_select_returns_native_for_registered_model(
    clean_viscosity_registry: None,
) -> None:
    cross = Model("CrossPowerLaw").register_with(viscosityModel)

    selected = select_viscosity_model(_cfg("CrossPowerLaw"))

    assert isinstance(selected, ModelSpec)
    assert selected is cross


def test_select_falls_back_to_openfoam_for_unknown_model() -> None:
    factory = MagicMock()
    selected = select_viscosity_model(_cfg("BirdCarreau"), of_factory=factory)
    assert isinstance(selected, OpenFOAMViscosityModel)
