# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""Tests for turbulence models.

NOTE: This file has been split into multiple focused test files:
- test_plugin_system.py: Plugin registration tests
- test_model_creation.py: Model instantiation tests
- test_model_setup.py: Setup and initialization tests
- test_file_reading.py: File parsing tests
- test_operations.py: Operations collection tests
- test_integration.py: Integration tests with real OpenFOAM cases

This file is kept for backward compatibility but may be removed in the future.
Use the individual test files for new tests.
"""

import pytest
from pydantic import ValidationError
from unittest.mock import MagicMock, patch

from foamadapter.turbulence import (
    TurbulenceModel,
    kOmegaSSTModel,
    kEpsilonModel,
    SmagorinskyModel,
    LaminarModel,
)
from foamadapter.core.plugin_system import PluginSystem


# ============================================================================
# Plugin System Tests
# ============================================================================


def test_turbulence_plugin_registration():
    """Test that turbulence models are registered in PluginSystem."""
    registry = PluginSystem.get_registered("TurbulenceModel")
    assert registry is not None

    plugin_names = [cls.__name__ for cls in registry.plugin_registry]
    assert "kOmegaSSTModel" in plugin_names
    assert "SmagorinskyModel" in plugin_names
    assert "LaminarModel" in plugin_names


def test_turbulence_json_schema():
    """Test that turbulence model generates valid JSON schema."""
    Model = TurbulenceModel.plugin_model
    schema = Model.model_json_schema()

    assert "properties" in schema
    assert "config" in schema["properties"]

    discriminator = schema["properties"]["config"]["discriminator"]
    assert discriminator["propertyName"] == "model_key"

    mapping = discriminator["mapping"]
    assert "RAS_kOmegaSST" in mapping
    assert "RAS_kEpsilon" in mapping
    assert "LES_Smagorinsky" in mapping
    assert "laminar" in mapping


# ============================================================================
# Model Creation Tests
# ============================================================================


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


# ============================================================================
# Model Properties Tests
# ============================================================================


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


# ============================================================================
# Instance Creation Tests
# ============================================================================


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


# ============================================================================
# File Reading Tests
# ============================================================================


@pytest.fixture
def mock_pybfoam_dict():
    """Create mock OpenFOAM dictionary."""
    mock_dict = MagicMock()
    return mock_dict


def test_from_file_ras(mock_pybfoam_dict):
    """Test reading RAS configuration from file."""
    mock_dict = mock_pybfoam_dict

    # Mock the dictionary structure for pybFoam's API
    mock_ras_dict = MagicMock()

    # Mock get[str] syntax - create a mock object that supports subscript
    mock_get = MagicMock()
    mock_ras_get = MagicMock()

    def mock_get_str(key):
        if key == "simulationType":
            return "RAS"
        elif key == "RASModel":
            return "kOmegaSST"
        return None

    # Set up subscript behavior: get[str] returns the getter function
    mock_get.__getitem__ = lambda self, t: mock_get_str
    mock_ras_get.__getitem__ = lambda self, t: mock_get_str

    mock_dict.get = mock_get
    mock_dict.subDict = lambda name: mock_ras_dict if name == "RAS" else None
    mock_ras_dict.get = mock_ras_get

    with patch("pybFoam.dictionary.read", return_value=mock_dict):
        wrapper = TurbulenceModel.from_file("constant/turbulenceProperties")
        model = wrapper.config  # Access the actual model from the wrapper

    assert isinstance(model, kOmegaSSTModel)
    assert model.model_key == "RAS_kOmegaSST"
    assert model.simulation_type == "RAS"
    assert model.turb_model_type == "kOmegaSST"


# ============================================================================
# Operations Tests
# ============================================================================


def test_ras_model_operations():
    """Test that RAS model has operations."""
    config = {
        "model_key": "RAS_kOmegaSST",
        "simulation_type": "RAS",
        "turb_model_type": "kOmegaSST",
    }
    wrapper = TurbulenceModel.plugin_model(config=config)
    model = wrapper.config  # Access the actual model from the wrapper

    ops = model.operations()
    assert ops is not None


def test_laminar_model_operations():
    """Test that laminar model has empty operations."""
    config = {
        "model_key": "laminar",
        "simulation_type": "laminar",
        "turb_model_type": None,
    }
    wrapper = TurbulenceModel.plugin_model(config=config)
    model = wrapper.config  # Access the actual model from the wrapper

    ops = model.operations()
    assert ops is not None
    # Laminar should have no operations


# ============================================================================
# Real Test Case Tests
# ============================================================================


def test_read_real_turbulence_case():
    """Test reading turbulence properties from real OpenFOAM test case."""
    from pathlib import Path

    # Get path to test case
    test_dir = Path(__file__).parent
    turb_case = test_dir / "turb_case" / "constant" / "turbulenceProperties"

    # Skip test if case doesn't exist
    if not turb_case.exists():
        pytest.skip(f"Test case not found: {turb_case}")

    # Real case uses kEpsilon which now works with composite key
    wrapper = TurbulenceModel.from_file(str(turb_case))
    model = wrapper.config

    assert isinstance(model, kEpsilonModel)
    assert model.model_key == "RAS_kEpsilon"
    assert model.simulation_type == "RAS"
    assert model.turb_model_type == "kEpsilon"
