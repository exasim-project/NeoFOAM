# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Tests for simple IO decorator patterns (no subdicts).

Demonstrates:
- Loading simple configs from YAML and JSON
- Writing simple configs to YAML and JSON
"""

import pytest
from pathlib import Path

from foamadapter.io import (
    BaseConfig,
    YAML,
    JSON,
    IOStrategy,
)


# ============================================================================
# Config Classes
# ============================================================================

@IOStrategy(YAML("simple.yaml"))
class SimpleYAMLConfig(BaseConfig):
    name: str
    value: int
    enabled: bool


@IOStrategy(JSON("simple.json"))
class SimpleJSONConfig(BaseConfig):
    name: str
    value: int
    enabled: bool


# ============================================================================
# Tests
# ============================================================================

@pytest.mark.parametrize("config_class", [
    SimpleYAMLConfig,
    SimpleJSONConfig,
])
def test_load_simple(io_fixtures, config_class):
    """Test loading simple config from fixture (YAML and JSON)."""
    loaded = config_class.load(io_fixtures)
    
    assert loaded.name == "test"
    assert loaded.value == 42
    assert loaded.enabled is True


@pytest.mark.parametrize("config_class,filename", [
    (SimpleYAMLConfig, "output.yaml"),
    (SimpleJSONConfig, "output.json"),
])
def test_write_simple(tmp_path, config_class, filename):
    """Test writing simple config to file (YAML and JSON)."""
    config = config_class(name="demo", value=100, enabled=False)
    
    # Write to file
    output_file = tmp_path / filename
    config.save(output_file)
    
    assert output_file.exists()
    
    # Verify by loading back directly from the file
    loaded = config_class.load(output_file)
    assert loaded.name == "demo"
    assert loaded.value == 100
    assert loaded.enabled is False

