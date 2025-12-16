# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

"""Tests for Pydantic configuration validation."""

import pytest
from .initialization_test_models import (
    TurbulenceConfig,
    TransportConfig,
    SolverConfig,
)


def test_pydantic_config_validation():
    """Test that Pydantic validates configuration."""
    # Valid config
    config = TurbulenceConfig(model_type="kOmega", wall_function=False)
    assert config.model_type == "kOmega"

    # Default values
    default_config = TurbulenceConfig()
    assert default_config.model_type == "kEpsilon"
    assert default_config.wall_function is True


def test_solver_config_defaults():
    """Test solver configuration defaults."""
    config = SolverConfig()

    assert config.max_iterations == 100
    assert config.tolerance == 1e-6
    assert config.time_step == 0.001


def test_transport_config_validation():
    """Test transport config validates positive values."""
    # Valid config
    config = TransportConfig(viscosity=1e-5, density=1.2)
    assert config.viscosity == 1e-5

    # Invalid viscosity (should fail)
    with pytest.raises(Exception):
        TransportConfig(viscosity=-1.0)

    # Invalid density (should fail)
    with pytest.raises(Exception):
        TransportConfig(density=0)
