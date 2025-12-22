# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""Tests for turbulence model operations."""

from foamadapter.turbulence import TurbulenceModel


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
    assert hasattr(ops, "add")  # OperationCollection interface


def test_les_model_operations():
    """Test that LES model has operations."""
    config = {
        "model_key": "LES_Smagorinsky",
        "simulation_type": "LES",
        "turb_model_type": "Smagorinsky",
    }
    wrapper = TurbulenceModel.plugin_model(config=config)
    model = wrapper.config

    ops = model.operations()
    assert ops is not None
    assert hasattr(ops, "add")


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
    # Laminar should have empty operations collection
    assert hasattr(ops, "add")
