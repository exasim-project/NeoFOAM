# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Tests for simple IO decorator patterns (no subdicts).

Demonstrates:
- Loading simple configs from YAML and JSON
- Writing simple configs to YAML and JSON
- Validation error reporting via load(validate=True/False)
- FileNotFoundError on missing file
"""

from pathlib import Path
from typing import cast, Any

import pytest

from pydantic import ValidationError, Field
from neofoam.io import (
    BaseConfig,
    YAML,
    JSON,
    IOStrategy,
)


@IOStrategy(YAML("simple.yaml"))
class SimpleYAMLConfig(BaseConfig):
    identifier: str
    count: int = Field(gt=0)
    active: bool
    percentage: float = Field(ge=0, le=100)


@IOStrategy(JSON("simple.json"))
class SimpleJSONConfig(BaseConfig):
    identifier: str
    count: int = Field(gt=0)
    active: bool
    percentage: float = Field(ge=0, le=100)


@pytest.mark.parametrize(
    "config_class",
    [
        SimpleYAMLConfig,
        SimpleJSONConfig,
    ],
)
def test_load_simple(io_fixtures: Path, config_class: type[BaseConfig]) -> None:
    """Test loading simple config from fixture (YAML and JSON)."""
    loaded = cast(Any, config_class.load(case_dir=io_fixtures))

    assert loaded.identifier == "test_config"
    assert loaded.count == 42
    assert loaded.active is True
    assert loaded.percentage == 75.5


@pytest.mark.parametrize(
    "config_class,filename",
    [
        (SimpleYAMLConfig, "output.yaml"),
        (SimpleJSONConfig, "output.json"),
    ],
)
def test_write_simple(
    tmp_path: Path, config_class: type[BaseConfig], filename: str
) -> None:
    """Test writing simple config to file (YAML and JSON)."""
    config = config_class(identifier="demo", count=100, active=False, percentage=50.0)

    # Write to file
    output_file = tmp_path / filename
    config.save(case_dir=output_file)

    assert output_file.exists()

    # Verify by loading back directly from the file
    loaded = cast(Any, config_class.load(case_dir=output_file))
    assert loaded.identifier == "demo"
    assert loaded.count == 100
    assert loaded.active is False
    assert loaded.percentage == 50.0


@pytest.mark.parametrize(
    "config_class,filename",
    [
        (SimpleYAMLConfig, "simple.yaml"),
        (SimpleJSONConfig, "simple.json"),
    ],
)
def test_write_and_reload_identical(
    io_fixtures: Path, tmp_path: Path, config_class: type[BaseConfig], filename: str
) -> None:
    """Test that writing a loaded config to a new file produces identical data.

    Loads from the fixture, writes to a new file, reloads from the new file,
    and asserts both instances carry the same data.
    """
    original = config_class.load(case_dir=io_fixtures)
    original.save(case_dir=tmp_path, file=filename)

    reloaded = config_class.load(case_dir=tmp_path, file=filename)
    assert reloaded.model_dump() == original.model_dump()


@pytest.mark.parametrize(
    "config_class,invalid_file",
    [
        (SimpleYAMLConfig, "invalid_simple.yaml"),
        (SimpleJSONConfig, "invalid_simple.json"),
    ],
)
def test_validation_error_wrong_type(
    io_fixtures: Path, config_class: type[BaseConfig], invalid_file: str
) -> None:
    """Test that validation correctly identifies wrong field types and validator violations.

    The invalid configs have 'count' as a string instead of int,
    and percentage=150 (violates le=100 constraint).
    load(validate=False) bypasses validation so all data is available for inspection.
    load(validate=True) raises ValidationError with all errors.
    """
    # Load without validation - all data is available for inspection
    loaded_data = cast(
        Any, config_class.load(case_dir=io_fixtures, validate=False, file=invalid_file)
    )
    assert loaded_data.identifier == "test_config"
    assert loaded_data.count == "not_an_integer"  # Wrong type loaded
    assert loaded_data.active is True
    assert loaded_data.percentage == 150  # Invalid value loaded

    # Load with validation - raises ValidationError with all errors
    with pytest.raises(ValidationError) as exc_info:
        config_class.load(case_dir=io_fixtures, validate=True, file=invalid_file)

    # Verify the error details
    errors = exc_info.value.errors()
    assert len(errors) == 2

    # Check for count type error
    int_errors = [e for e in errors if e["loc"] == ("count",)]
    assert len(int_errors) == 1
    assert int_errors[0]["type"] == "int_parsing"
    assert "not_an_integer" in str(int_errors[0]["input"])

    # Check for percentage constraint violation
    le_errors = [e for e in errors if e["loc"] == ("percentage",)]
    assert len(le_errors) == 1
    assert le_errors[0]["type"] == "less_than_equal"
    assert le_errors[0]["ctx"]["le"] == 100


@pytest.mark.parametrize(
    "config_class,missing_file",
    [
        (SimpleYAMLConfig, "nonexistent.yaml"),
        (SimpleJSONConfig, "nonexistent.json"),
    ],
)
def test_load_missing_file_raises(
    tmp_path: Path, config_class: type[BaseConfig], missing_file: str
) -> None:
    """Loading from a non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        config_class.load(case_dir=tmp_path, file=missing_file)
