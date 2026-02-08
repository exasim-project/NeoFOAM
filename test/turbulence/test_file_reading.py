# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""Tests for reading turbulence configuration from OpenFOAM files."""

import pytest

pytestmark = pytest.mark.skip(
    reason="Outdated - framework refactored (@Model.build decorator removed) - Old API imports removed"
)

# from unittest.mock import MagicMock, patch
#
# from foamadapter.turbulence import TurbulenceModel, kOmegaSSTModel


@pytest.fixture
def mock_pybfoam_dict():
    """Create mock OpenFOAM dictionary."""
    mock_dict = MagicMock()
    return mock_dict


def test_from_file_ras_komegasst(mock_pybfoam_dict):
    """Test reading RAS k-omega SST configuration from file."""
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


def test_from_file_les(mock_pybfoam_dict):
    """Test reading LES configuration from file."""
    mock_dict = mock_pybfoam_dict
    mock_les_dict = MagicMock()

    mock_get = MagicMock()
    mock_les_get = MagicMock()

    def mock_get_str(key):
        if key == "simulationType":
            return "LES"
        elif key == "LESModel":
            return "Smagorinsky"
        return None

    mock_get.__getitem__ = lambda self, t: mock_get_str
    mock_les_get.__getitem__ = lambda self, t: mock_get_str

    mock_dict.get = mock_get
    mock_dict.subDict = lambda name: mock_les_dict if name == "LES" else None
    mock_les_dict.get = mock_les_get

    with patch("pybFoam.dictionary.read", return_value=mock_dict):
        wrapper = TurbulenceModel.from_file("constant/turbulenceProperties")
        model = wrapper.config

    assert model.model_key == "LES_Smagorinsky"
    assert model.simulation_type == "LES"
    assert model.turb_model_type == "Smagorinsky"


def test_from_file_laminar(mock_pybfoam_dict):
    """Test reading laminar configuration from file."""
    mock_dict = mock_pybfoam_dict

    mock_get = MagicMock()

    def mock_get_str(key):
        if key == "simulationType":
            return "laminar"
        return None

    mock_get.__getitem__ = lambda self, t: mock_get_str
    mock_dict.get = mock_get

    with patch("pybFoam.dictionary.read", return_value=mock_dict):
        wrapper = TurbulenceModel.from_file("constant/turbulenceProperties")
        model = wrapper.config

    assert model.model_key == "laminar"
    assert model.simulation_type == "laminar"
    assert model.turb_model_type is None
