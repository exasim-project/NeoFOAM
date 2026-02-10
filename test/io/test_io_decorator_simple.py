# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Tests for simple IO decorator patterns (no subdicts).

Demonstrates:
- Loading simple configs from YAML and JSON
- Writing simple configs to YAML and JSON
"""

import pytest

from pydantic import ValidationError
from neofoam.io import (
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


@pytest.mark.parametrize(
    "config_class",
    [
        SimpleYAMLConfig,
        SimpleJSONConfig,
    ],
)
def test_load_simple(io_fixtures, config_class):
    """Test loading simple config from fixture (YAML and JSON)."""
    loaded = config_class.load(io_fixtures)

    assert loaded.name == "test"
    assert loaded.value == 42
    assert loaded.enabled is True


@pytest.mark.parametrize(
    "config_class,filename",
    [
        (SimpleYAMLConfig, "output.yaml"),
        (SimpleJSONConfig, "output.json"),
    ],
)
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


# ============================================================================
# Error Validation Tests
# ============================================================================


@pytest.mark.parametrize(
    "format,config_file",
    [
        ("yaml", "invalid_simple.yaml"),
        ("json", "invalid_simple.json"),
    ],
)
def test_validation_error_wrong_type(io_fixtures, format, config_file):
    """Test that validation correctly identifies wrong field types.

    The invalid configs have 'value' as a string instead of int.
    Uses model_construct to load without validation, then model_validate to check errors.
    """
    # Create config class dynamically for the invalid file
    if format == "yaml":

        @IOStrategy(YAML(config_file))
        class InvalidSimpleConfig(BaseConfig):
            name: str
            value: int
            enabled: bool
    else:

        @IOStrategy(JSON(config_file))
        class InvalidSimpleConfig(BaseConfig):
            name: str
            value: int
            enabled: bool

    # Load with model_construct (bypasses validation)
    loaded_data = InvalidSimpleConfig.load(io_fixtures)

    # Verify data was loaded (with wrong type)
    assert loaded_data.name == "test"
    assert loaded_data.value == "not_an_integer"  # Wrong type loaded
    assert loaded_data.enabled is True

    # Now validate - should raise ValidationError for wrong type
    with pytest.raises(ValidationError) as exc_info:
        InvalidSimpleConfig.model_validate(loaded_data.model_dump())

    # Verify the error details
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["type"] == "int_parsing"
    assert errors[0]["loc"] == ("value",)
    assert "not_an_integer" in str(errors[0]["input"])
