# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""Tests for turbulence model setup and initialization."""

from unittest.mock import MagicMock, patch

from foamadapter.turbulence import TurbulenceModel


@patch("foamadapter.turbulence.models.incompressibleTurbulenceModel")
def test_ras_model_setup(mock_turbulence):
    """Test RAS model setup and instance creation via DAG."""
    config = {
        "model_key": "RAS_kOmegaSST",
        "simulation_type": "RAS",
        "turb_model_type": "kOmegaSST",
    }
    wrapper = TurbulenceModel.plugin_model(config=config)
    model = wrapper.config  # Access the actual model from the wrapper

    # Mock mesh and setup
    mock_mesh = MagicMock()
    lazy_inits = model.setup(mock_mesh)

    # Verify LazyInit returned
    assert len(lazy_inits) == 1
    assert lazy_inits[0].name == "models.turbulence"
    assert "fields.U" in lazy_inits[0].depends_on
    assert "fields.phi" in lazy_inits[0].depends_on
    assert "fields.laminarTransport" in lazy_inits[0].depends_on


@patch("foamadapter.turbulence.models.incompressibleTurbulenceModel")
def test_les_model_setup(mock_turbulence):
    """Test LES model setup and instance creation via DAG."""
    config = {
        "model_key": "LES_Smagorinsky",
        "simulation_type": "LES",
        "turb_model_type": "Smagorinsky",
    }
    wrapper = TurbulenceModel.plugin_model(config=config)
    model = wrapper.config

    mock_mesh = MagicMock()
    lazy_inits = model.setup(mock_mesh)

    assert len(lazy_inits) == 1
    assert lazy_inits[0].name == "models.turbulence"
    assert "fields.U" in lazy_inits[0].depends_on
    assert "fields.phi" in lazy_inits[0].depends_on
    assert "fields.laminarTransport" in lazy_inits[0].depends_on


@patch("foamadapter.turbulence.models.incompressibleTurbulenceModel")
def test_laminar_model_setup(mock_turbulence):
    """Test laminar model setup."""
    config = {
        "model_key": "laminar",
        "simulation_type": "laminar",
        "turb_model_type": None,
    }
    wrapper = TurbulenceModel.plugin_model(config=config)
    model = wrapper.config

    mock_mesh = MagicMock()
    lazy_inits = model.setup(mock_mesh)

    # Laminar model still creates turbulence instance for compatibility
    assert len(lazy_inits) == 1
    assert lazy_inits[0].name == "models.turbulence"
