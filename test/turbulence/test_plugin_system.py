# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""Tests for turbulence model plugin system registration."""

import pytest

pytestmark = pytest.mark.skip(
    reason="Outdated - framework refactored (@Model.build decorator removed)"
)

from foamadapter.turbulence import (
    TurbulenceModel,
    kOmegaSSTModel,
    kEpsilonModel,
    SmagorinskyModel,
    LaminarModel,
)
from foamadapter.core.plugin_system import PluginSystem


def test_turbulence_plugin_registration():
    """Test that turbulence models are registered in PluginSystem.

    Note: This test checks foamadapter.turbulence models.
    There's also foamadapter.models.turbulence used by the solver.
    """
    # Ensure models are imported (triggers @register decorators)
    _ = (kOmegaSSTModel, kEpsilonModel, SmagorinskyModel, LaminarModel)

    registry = PluginSystem.get_registered("TurbulenceModel")
    assert registry is not None

    plugin_names = [cls.__name__ for cls in registry.plugin_registry]

    # Check if we have the expected models (may vary based on import order)
    has_detailed_models = "kOmegaSSTModel" in plugin_names
    has_wrapper_models = "OpenFOAMTurbulence" in plugin_names

    # Either implementation should be present
    assert has_detailed_models or has_wrapper_models, (
        f"No turbulence models found. Got: {plugin_names}"
    )

    if has_detailed_models:
        assert "kEpsilonModel" in plugin_names
        assert "SmagorinskyModel" in plugin_names

    assert "LaminarModel" in plugin_names


def test_turbulence_json_schema():
    """Test that turbulence model generates valid JSON schema.

    Schema should use model_key as discriminator with composite keys like
    'RAS_kOmegaSST', 'RAS_kEpsilon', 'LES_Smagorinsky', 'laminar'.
    """
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
