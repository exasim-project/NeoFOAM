# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Tests for nested subdict IO patterns.

Demonstrates:
- Loading from flat subdicts (top-level keys)
- Loading from nested subdicts (dot notation)
- Subdict isolation (each config only sees its own section)
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


@IOStrategy(YAML("nested.yaml", subdict="metadata"))
class MetadataYAMLConfig(BaseConfig):
    name: str
    version: str
    priority: int = Field(gt=0, description="Priority must be positive")


@IOStrategy(YAML("nested.yaml", subdict="processing.stage1"))
class Stage1YAMLConfig(BaseConfig):
    algorithm: str
    threshold: float = Field(gt=0)
    maxIterations: int = Field(gt=0, le=1000)
    batchSize: int = Field(gt=0)


@IOStrategy(YAML("nested.yaml", subdict="processing.stage2"))
class Stage2YAMLConfig(BaseConfig):
    algorithm: str
    threshold: float = Field(gt=0)
    maxIterations: int = Field(gt=0, le=1000)
    batchSize: int = Field(gt=0)


# -- JSON models -----------------------------------------------------------


@IOStrategy(JSON("nested.json", subdict="metadata"))
class MetadataJSONConfig(BaseConfig):
    name: str
    version: str
    priority: int = Field(gt=0, description="Priority must be positive")


@IOStrategy(JSON("nested.json", subdict="processing.stage1"))
class Stage1JSONConfig(BaseConfig):
    algorithm: str
    threshold: float = Field(gt=0)
    maxIterations: int = Field(gt=0, le=1000)
    batchSize: int = Field(gt=0)


@IOStrategy(JSON("nested.json", subdict="processing.stage2"))
class Stage2JSONConfig(BaseConfig):
    algorithm: str
    threshold: float = Field(gt=0)
    maxIterations: int = Field(gt=0, le=1000)
    batchSize: int = Field(gt=0)


# -- OpenFOAM models -------------------------------------------------------


@IOStrategy(OF("nested.of", subdict="metadata"))
class MetadataOpenFOAMConfig(BaseConfig):
    name: str
    version: float
    priority: int = Field(gt=0)


@IOStrategy(OF("nested.of", subdict="processing.stage1"))
class Stage1OpenFOAMConfig(BaseConfig):
    algorithm: str
    threshold: float = Field(gt=0)
    maxIterations: int = Field(gt=0, le=1000)
    batchSize: int = Field(gt=0)


@IOStrategy(OF("nested.of", subdict="processing.stage2"))
class Stage2OpenFOAMConfig(BaseConfig):
    algorithm: str
    threshold: float = Field(gt=0)
    maxIterations: int = Field(gt=0, le=1000)
    batchSize: int = Field(gt=0)


# -- Tests ------------------------------------------------------------------


@pytest.mark.parametrize(
    "metadata_class,stage1_class,stage2_class",
    [
        (MetadataYAMLConfig, Stage1YAMLConfig, Stage2YAMLConfig),
        (MetadataJSONConfig, Stage1JSONConfig, Stage2JSONConfig),
        (MetadataOpenFOAMConfig, Stage1OpenFOAMConfig, Stage2OpenFOAMConfig),
    ],
)
def test_load_nested(io_fixtures, metadata_class, stage1_class, stage2_class):
    """Test loading multiple nested subdicts from fixture (YAML, JSON and OpenFOAM).

    Demonstrates:
    - Flat subdict: metadata (top-level key)
    - Nested subdict: processing.stage1 (2-level nesting)
    - Nested subdict: processing.stage2 (2-level nesting, different key)
    - Subdict isolation (each config only sees its own section)
    - Field validators (gt=0, le=1000)
    """
    # Load each subdict independently
    metadata = metadata_class.load(case_dir=io_fixtures)
    stage1 = stage1_class.load(case_dir=io_fixtures)
    stage2 = stage2_class.load(case_dir=io_fixtures)

    # Verify isolation - each config only sees its subdict
    assert metadata.name == "TestApp"
    assert float(metadata.version) == 1.0  # str for YAML/JSON, float for OF
    assert metadata.priority == 5

    assert stage1.algorithm == "fast"
    assert stage1.threshold == 0.001
    assert stage1.maxIterations == 100
    assert stage1.batchSize == 32

    assert stage2.algorithm == "accurate"
    assert stage2.threshold == 0.0001
    assert stage2.maxIterations == 50
    assert stage2.batchSize == 16


@pytest.mark.parametrize(
    "metadata_class,stage1_class,filename",
    [
        (MetadataYAMLConfig, Stage1YAMLConfig, "written.yaml"),
        (MetadataJSONConfig, Stage1JSONConfig, "written.json"),
    ],
)
def test_write_and_reload_identical(tmp_path, metadata_class, stage1_class, filename):
    """Test that configs constructed in code survive a write/reload round-trip.

    Creates config instances directly, writes them to a fresh file in tmp_path,
    reloads from the file, compares model_dump() outputs, then removes the file.
    """
    original_meta = metadata_class(name="MyApp", version="2.0", priority=10)
    original_s1 = stage1_class(
        algorithm="balanced", threshold=0.01, maxIterations=200, batchSize=64
    )

    # Write both subdicts to a new file
    original_meta.save(case_dir=tmp_path, file=filename)
    original_s1.save(case_dir=tmp_path, file=filename)

    # Reload from the new file and compare
    reloaded_meta = metadata_class.load(case_dir=tmp_path, file=filename)
    reloaded_s1 = stage1_class.load(case_dir=tmp_path, file=filename)

    assert reloaded_meta.model_dump() == original_meta.model_dump()
    assert reloaded_s1.model_dump() == original_s1.model_dump()

    # Clean up the written file
    (tmp_path / filename).unlink()


@pytest.mark.parametrize(
    "config_class,invalid_file",
    [
        (MetadataYAMLConfig, "invalid_nested.yaml"),
        (MetadataJSONConfig, "invalid_nested.json"),
        (MetadataOpenFOAMConfig, "invalid_nested.of"),
    ],
)
def test_validation_error_missing_field(io_fixtures, config_class, invalid_file):
    """Test that validation correctly identifies missing required fields and validator violations.

    The invalid configs are missing the 'version' field and have priority=0 (violates gt=0).
    load(validate=False) bypasses validation so all data is available for inspection.
    load(validate=True) raises ValidationError with all errors.
    """
    # Load without validation - all data is available for inspection
    loaded_data = config_class.load(
        case_dir=io_fixtures, validate=False, file=invalid_file
    )
    assert loaded_data.name == "TestApp"
    assert loaded_data.priority == 0  # Invalid value loaded without validation

    # Load with validation - raises ValidationError with all errors
    with pytest.raises(ValidationError) as exc_info:
        config_class.load(case_dir=io_fixtures, validate=True, file=invalid_file)

    # Verify the error details
    errors = exc_info.value.errors()
    assert len(errors) == 2

    # Check for missing version field
    missing_errors = [e for e in errors if e["type"] == "missing"]
    assert len(missing_errors) == 1
    assert missing_errors[0]["loc"] == ("version",)

    # Check for priority constraint violation
    gt_errors = [e for e in errors if e["type"] == "greater_than"]
    assert len(gt_errors) == 1
    assert gt_errors[0]["loc"] == ("priority",)
    assert gt_errors[0]["ctx"]["gt"] == 0


@pytest.mark.parametrize(
    "config_class,missing_file",
    [
        (MetadataYAMLConfig, "nonexistent.yaml"),
        (MetadataJSONConfig, "nonexistent.json"),
        (MetadataOpenFOAMConfig, "nonexistent.of"),
    ],
)
def test_load_missing_file_raises(tmp_path, config_class, missing_file):
    """Loading from a non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        config_class.load(case_dir=tmp_path, file=missing_file)


@pytest.mark.parametrize(
    "config_class,config_file",
    [
        (MetadataYAMLConfig, "simple.yaml"),
        (MetadataJSONConfig, "simple.json"),
        (MetadataOpenFOAMConfig, "simple.of"),
    ],
)
def test_load_missing_subdict_raises(io_fixtures, config_class, config_file):
    """Loading from an existing file but wrong subdict path raises KeyError.

    Uses a file that exists but doesn't contain the 'metadata' subdict.
    """
    with pytest.raises(KeyError, match="not found"):
        config_class.load(case_dir=io_fixtures, file=config_file)


@pytest.mark.parametrize(
    "metadata_class,stage1_class,filename",
    [
        (MetadataYAMLConfig, Stage1YAMLConfig, "nested.yaml"),
        (MetadataJSONConfig, Stage1JSONConfig, "nested.json"),
        (MetadataOpenFOAMConfig, Stage1OpenFOAMConfig, "nested.of"),
    ],
)
def test_write_nested(temp_fixture_copy, metadata_class, stage1_class, filename):
    """Test that saving a loaded subdict produces identical data on reload."""
    test_file = temp_fixture_copy(filename)
    test_dir = test_file.parent

    original = stage1_class.load(case_dir=test_dir)
    original.save(case_dir=test_dir)
    reloaded = stage1_class.load(case_dir=test_dir)

    assert reloaded == original

    # Verify other subdicts are preserved
    meta = metadata_class.load(case_dir=test_dir)
    assert meta.name == "TestApp"
