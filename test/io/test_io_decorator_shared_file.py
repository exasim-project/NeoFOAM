# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Tests for shared file preservation patterns.

Demonstrates:
- Multiple models reading from same file with subdict isolation
- Writing one subdict preserves other sections (critical for shared config files)
- Validation error reporting via load(validate=True/False)
- FileNotFoundError / KeyError on missing file or subdict
"""

import pytest

from pydantic import ValidationError, Field
from neofoam.io import (
    BaseConfig,
    YAML,
    JSON,
    OF,
    IOStrategy,
)


# -- YAML models -----------------------------------------------------------


@IOStrategy(YAML("shared.yaml", subdict="config.service_a"))
class ServiceAYAMLConfig(BaseConfig):
    timeout: int = Field(gt=0)
    maxConnections: int = Field(gt=0, le=100)


@IOStrategy(YAML("shared.yaml", subdict="config.service_b"))
class ServiceBYAMLConfig(BaseConfig):
    endpoint: str
    port: int = Field(gt=0, le=65535)
    poolSize: int = Field(gt=0)


@IOStrategy(YAML("shared.yaml", subdict="config.service_c"))
class ServiceCYAMLConfig(BaseConfig):
    enabled: bool
    bufferSize: int = Field(gt=0)


# -- JSON models -----------------------------------------------------------


@IOStrategy(JSON("shared.json", subdict="config.service_a"))
class ServiceAJSONConfig(BaseConfig):
    timeout: int = Field(gt=0)
    maxConnections: int = Field(gt=0, le=100)


@IOStrategy(JSON("shared.json", subdict="config.service_b"))
class ServiceBJSONConfig(BaseConfig):
    endpoint: str
    port: int = Field(gt=0, le=65535)
    poolSize: int = Field(gt=0)


@IOStrategy(JSON("shared.json", subdict="config.service_c"))
class ServiceCJSONConfig(BaseConfig):
    enabled: bool
    bufferSize: int = Field(gt=0)


# -- OpenFOAM models -------------------------------------------------------


@IOStrategy(OF("shared.of", subdict="config.service_a"))
class ServiceAOpenFOAMConfig(BaseConfig):
    timeout: int = Field(gt=0)
    maxConnections: int = Field(gt=0, le=100)


@IOStrategy(OF("shared.of", subdict="config.service_b"))
class ServiceBOpenFOAMConfig(BaseConfig):
    endpoint: str
    port: int = Field(gt=0, le=65535)
    poolSize: int = Field(gt=0)


@IOStrategy(OF("shared.of", subdict="config.service_c"))
class ServiceCOpenFOAMConfig(BaseConfig):
    enabled: bool
    bufferSize: int = Field(gt=0)


# -- Tests ------------------------------------------------------------------


@pytest.mark.parametrize(
    "service_a_class,service_b_class,service_c_class,filename",
    [
        (ServiceAYAMLConfig, ServiceBYAMLConfig, ServiceCYAMLConfig, "shared.yaml"),
        (ServiceAJSONConfig, ServiceBJSONConfig, ServiceCJSONConfig, "shared.json"),
        (
            ServiceAOpenFOAMConfig,
            ServiceBOpenFOAMConfig,
            ServiceCOpenFOAMConfig,
            "shared.of",
        ),
    ],
)
def test_write_preserves_other_subdicts(
    temp_fixture_copy, service_a_class, service_b_class, service_c_class, filename
):
    """Test that writing one subdict preserves other sections in same file.

    Demonstrates partial file updates - critical for shared config files where
    multiple models manage different sections of the same file.
    Also tests field validators (gt=0, le constraints).
    """
    test_file = temp_fixture_copy(filename)
    test_dir = test_file.parent
    timeout = 60

    # Load original values from directory
    original_b = service_b_class.load(case_dir=test_dir)
    original_c = service_c_class.load(case_dir=test_dir)
    assert original_b.endpoint == "example.com"
    assert original_c.enabled is True

    # Update only service_a section
    service_a = service_a_class.load(case_dir=test_dir)
    service_a.timeout = timeout
    service_a.save(case_dir=test_dir)

    # Verify service_a changed but other sections unchanged
    updated_a = service_a_class.load(case_dir=test_dir)
    updated_b = service_b_class.load(case_dir=test_dir)
    updated_c = service_c_class.load(case_dir=test_dir)

    assert updated_a.timeout == timeout
    assert updated_a.maxConnections == 10

    assert updated_b.endpoint == "example.com"
    assert updated_b.port == 8080

    assert updated_c.enabled is True
    assert updated_c.bufferSize == 1024


@pytest.mark.parametrize(
    "config_class,invalid_file",
    [
        (ServiceBYAMLConfig, "invalid_shared.yaml"),
        (ServiceBJSONConfig, "invalid_shared.json"),
        (ServiceBOpenFOAMConfig, "invalid_shared.of"),
    ],
)
def test_validation_error_missing_field_in_subdict(
    io_fixtures, config_class, invalid_file
):
    """Test that validation correctly identifies missing required fields and validator violations in subdicts.

    The invalid configs are missing the 'port' field and have poolSize=-5 (violates gt=0).
    load(validate=False) bypasses validation so all data is available for inspection.
    load(validate=True) raises ValidationError with all errors.
    """
    # Load without validation - all data is available for inspection
    loaded_data = config_class.load(
        case_dir=io_fixtures, validate=False, file=invalid_file
    )
    assert loaded_data.endpoint == "example.com"
    assert loaded_data.poolSize == -5  # Invalid value loaded

    # Load with validation - raises ValidationError with all errors
    with pytest.raises(ValidationError) as exc_info:
        config_class.load(case_dir=io_fixtures, validate=True, file=invalid_file)

    # Verify the error details
    errors = exc_info.value.errors()
    assert len(errors) == 2

    # Check for missing port field
    missing_errors = [e for e in errors if e["type"] == "missing"]
    assert len(missing_errors) == 1
    assert missing_errors[0]["loc"] == ("port",)

    # Check for poolSize constraint violation
    gt_errors = [e for e in errors if e["type"] == "greater_than"]
    assert len(gt_errors) == 1
    assert gt_errors[0]["loc"] == ("poolSize",)
    assert gt_errors[0]["ctx"]["gt"] == 0


@pytest.mark.parametrize(
    "config_class,missing_file",
    [
        (ServiceBYAMLConfig, "nonexistent.yaml"),
        (ServiceBJSONConfig, "nonexistent.json"),
        (ServiceBOpenFOAMConfig, "nonexistent.of"),
    ],
)
def test_load_missing_file_raises(tmp_path, config_class, missing_file):
    """Loading from a non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        config_class.load(case_dir=tmp_path, file=missing_file)


@pytest.mark.parametrize(
    "config_class,config_file",
    [
        (ServiceBYAMLConfig, "simple.yaml"),
        (ServiceBJSONConfig, "simple.json"),
        (ServiceBOpenFOAMConfig, "simple.of"),
    ],
)
def test_load_missing_subdict_raises(io_fixtures, config_class, config_file):
    """Loading from an existing file but wrong subdict path raises KeyError.

    Uses a file that exists but doesn't contain the 'config.service_b' subdict.
    """
    with pytest.raises(KeyError, match="not found"):
        config_class.load(case_dir=io_fixtures, file=config_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
