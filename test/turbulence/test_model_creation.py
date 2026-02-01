# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""Tests for turbulence model creation and instantiation."""

import pytest

pytestmark = pytest.mark.skip(
    reason="Outdated - framework refactored (@Model.build decorator removed)"
)

from pydantic import ValidationError

from foamadapter.turbulence import (
    TurbulenceModel,
    kOmegaSSTModel,
    SmagorinskyModel,
    LaminarModel,
)


def test_create_ras_model():
    """Test creating RAS turbulence model from config."""
    config = {
        "model_key": "RAS_kOmegaSST",
        "simulation_type": "RAS",
        "turb_model_type": "kOmegaSST",
    }
    wrapper = TurbulenceModel.plugin_model(config=config)
    model = wrapper.config  # Access the actual model from the wrapper

    assert isinstance(model, kOmegaSSTModel)
    assert model.model_key == "RAS_kOmegaSST"
    assert model.simulation_type == "RAS"
    assert model.turb_model_type == "kOmegaSST"


def test_create_les_model():
    """Test creating LES turbulence model from config."""
    config = {
        "model_key": "LES_Smagorinsky",
        "simulation_type": "LES",
        "turb_model_type": "Smagorinsky",
    }
    wrapper = TurbulenceModel.plugin_model(config=config)
    model = wrapper.config  # Access the actual model from the wrapper

    assert isinstance(model, SmagorinskyModel)
    assert model.model_key == "LES_Smagorinsky"
    assert model.simulation_type == "LES"
    assert model.turb_model_type == "Smagorinsky"


def test_create_laminar_model():
    """Test creating laminar turbulence model from config."""
    config = {
        "model_key": "laminar",
        "simulation_type": "laminar",
        "turb_model_type": None,
    }
    wrapper = TurbulenceModel.plugin_model(config=config)
    model = wrapper.config  # Access the actual model from the wrapper

    assert isinstance(model, LaminarModel)
    assert model.model_key == "laminar"
    assert model.simulation_type == "laminar"
    assert model.turb_model_type is None


def test_invalid_simulation_type():
    """Test that invalid simulation type raises ValidationError."""
    config = {
        "model_key": "INVALID",
        "simulation_type": "INVALID",
        "turb_model_type": None,
    }

    with pytest.raises(ValidationError):
        TurbulenceModel.plugin_model(config=config)


def test_turbulence_model_name():
    """Test that turbulence models have name property."""
    config = {
        "model_key": "RAS_kOmegaSST",
        "simulation_type": "RAS",
        "turb_model_type": "kOmegaSST",
    }
    wrapper = TurbulenceModel.plugin_model(config=config)
    model = wrapper.config  # Access the actual model from the wrapper

    assert hasattr(model, "name")
    assert model.name == "turbulence"
