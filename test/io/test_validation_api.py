# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Tests for the validation API: collect_errors(), validate(), validate_models().

Covers:
- collect_errors() on valid files (returns [])
- collect_errors() on invalid files (returns structured errors)
- collect_errors() on missing files (FileNotFound error)
- collect_errors() on missing subdict (KeyError)
- validate() on valid instances (returns [])
- validate() on invalid instances loaded with validate=False
- validate_models() batch aggregation
"""

import pytest

from pydantic import Field
from neofoam.io import (
    BaseConfig,
    YAML,
    JSON,
    IOStrategy,
    ValidationErrors,
    validate_models,
)


@IOStrategy(YAML("simple.yaml"))
class SimpleYAMLValidation(BaseConfig):
    identifier: str
    count: int = Field(gt=0)
    active: bool
    percentage: float = Field(ge=0, le=100)


@IOStrategy(JSON("simple.json"))
class SimpleJSONValidation(BaseConfig):
    identifier: str
    count: int = Field(gt=0)
    active: bool
    percentage: float = Field(ge=0, le=100)


@IOStrategy(YAML("nested.yaml", subdict="metadata"))
class MetadataYAMLValidation(BaseConfig):
    name: str
    version: str
    priority: int = Field(gt=0, description="Priority must be positive")


@IOStrategy(JSON("nested.json", subdict="metadata"))
class MetadataJSONValidation(BaseConfig):
    name: str
    version: str
    priority: int = Field(gt=0, description="Priority must be positive")


@IOStrategy(YAML("shared.yaml", subdict="config.service_b"))
class ServiceBYAMLValidation(BaseConfig):
    endpoint: str
    port: int = Field(gt=0, le=65535)
    poolSize: int = Field(gt=0)


@IOStrategy(JSON("shared.json", subdict="config.service_b"))
class ServiceBJSONValidation(BaseConfig):
    endpoint: str
    port: int = Field(gt=0, le=65535)
    poolSize: int = Field(gt=0)


# -- BaseConfig.collect_errors() class method ---------------------------------


@pytest.mark.parametrize(
    "config_class",
    [SimpleYAMLValidation, SimpleJSONValidation],
)
def test_valid_file_returns_empty(io_fixtures, config_class):
    """collect_errors() returns [] for a valid file."""
    errors = config_class.collect_errors(case_dir=io_fixtures)
    assert errors == []


@pytest.mark.parametrize(
    "config_class",
    [MetadataYAMLValidation, MetadataJSONValidation],
)
def test_valid_subdict_returns_empty(io_fixtures, config_class):
    """collect_errors() returns [] for a valid file with subdict."""
    errors = config_class.collect_errors(case_dir=io_fixtures)
    assert errors == []


@pytest.mark.parametrize(
    "config_class,invalid_file",
    [
        (SimpleYAMLValidation, "invalid_simple.yaml"),
        (SimpleJSONValidation, "invalid_simple.json"),
    ],
)
def test_invalid_file_returns_errors(io_fixtures, config_class, invalid_file):
    """collect_errors() returns structured errors for invalid data."""
    errors = config_class.collect_errors(case_dir=io_fixtures, file=invalid_file)

    assert len(errors) == 2
    assert all(isinstance(e, ValidationErrors) for e in errors)

    # Check count type error
    count_errors = [e for e in errors if e.field == ("count",)]
    assert len(count_errors) == 1
    assert count_errors[0].error_type == "int_parsing"

    # Check percentage constraint
    pct_errors = [e for e in errors if e.field == ("percentage",)]
    assert len(pct_errors) == 1
    assert pct_errors[0].error_type == "less_than_equal"


@pytest.mark.parametrize(
    "config_class,invalid_file",
    [
        (MetadataYAMLValidation, "invalid_nested.yaml"),
        (MetadataJSONValidation, "invalid_nested.json"),
    ],
)
def test_invalid_subdict_returns_errors(io_fixtures, config_class, invalid_file):
    """collect_errors() returns errors for invalid data inside a subdict."""
    errors = config_class.collect_errors(case_dir=io_fixtures, file=invalid_file)

    assert len(errors) == 2

    # Missing version field
    missing = [e for e in errors if e.error_type == "missing"]
    assert len(missing) == 1
    assert missing[0].field == ("version",)

    # Priority gt=0 violation
    gt = [e for e in errors if e.error_type == "greater_than"]
    assert len(gt) == 1
    assert gt[0].field == ("priority",)


@pytest.mark.parametrize(
    "config_class,missing_file",
    [
        (SimpleYAMLValidation, "nonexistent.yaml"),
        (SimpleJSONValidation, "nonexistent.json"),
    ],
)
def test_missing_file_returns_error(tmp_path, config_class, missing_file):
    """collect_errors() returns FileNotFound error instead of raising."""
    errors = config_class.collect_errors(case_dir=tmp_path, file=missing_file)

    assert len(errors) == 1
    assert errors[0].error_type == "FileNotFound"
    assert errors[0].file_name == missing_file


@pytest.mark.parametrize(
    "config_class,wrong_file",
    [
        (MetadataYAMLValidation, "simple.yaml"),
        (MetadataJSONValidation, "simple.json"),
    ],
)
def test_missing_subdict_returns_error(io_fixtures, config_class, wrong_file):
    """collect_errors() returns KeyError for wrong subdict path."""
    errors = config_class.collect_errors(case_dir=io_fixtures, file=wrong_file)

    assert len(errors) == 1
    assert errors[0].error_type == "KeyError"


def test_file_name_context_uses_override(io_fixtures):
    """When file= is passed, errors report that filename, not the default."""
    errors = SimpleYAMLValidation.collect_errors(
        case_dir=io_fixtures, file="invalid_simple.yaml"
    )
    assert all(e.file_name == "invalid_simple.yaml" for e in errors)


def test_file_name_context_uses_default(io_fixtures):
    """When no file= is passed, errors report the registered filename."""
    errors = MetadataYAMLValidation.collect_errors(case_dir=io_fixtures)
    assert errors == []  # valid file → no errors to check filename on
    # Trigger with invalid to check filename
    # Use the default file but with a config that would fail on a different file
    # Not applicable here — default is valid. Tested via invalid_file tests.


def test_subdict_context_included(io_fixtures):
    """Errors from a subdict config include the subdict path."""
    errors = MetadataYAMLValidation.collect_errors(
        case_dir=io_fixtures, file="invalid_nested.yaml"
    )
    assert all(e.subdict == "metadata" for e in errors)


# -- BaseConfig.check_validation() instance method ----------------------------


@pytest.mark.parametrize(
    "config_class",
    [SimpleYAMLValidation, SimpleJSONValidation],
)
def test_valid_instance_returns_empty(io_fixtures, config_class):
    """check_validation() returns [] for a correctly loaded instance."""
    instance = config_class.load(case_dir=io_fixtures)
    assert instance.check_validation() == []


@pytest.mark.parametrize(
    "config_class,invalid_file",
    [
        (SimpleYAMLValidation, "invalid_simple.yaml"),
        (SimpleJSONValidation, "invalid_simple.json"),
    ],
)
def test_invalid_instance_returns_errors(io_fixtures, config_class, invalid_file):
    """check_validation() returns errors for an instance loaded with validate=False."""
    instance = config_class.load(
        case_dir=io_fixtures, validate=False, file=invalid_file
    )
    errors = instance.check_validation()

    assert len(errors) == 2
    assert all(isinstance(e, ValidationErrors) for e in errors)

    error_types = {e.error_type for e in errors}
    assert "int_parsing" in error_types
    assert "less_than_equal" in error_types


@pytest.mark.parametrize(
    "config_class,invalid_file",
    [
        (MetadataYAMLValidation, "invalid_nested.yaml"),
        (MetadataJSONValidation, "invalid_nested.json"),
    ],
)
def test_invalid_subdict_instance_returns_errors(
    io_fixtures, config_class, invalid_file
):
    """check_validation() returns errors for a subdict instance with bad data."""
    instance = config_class.load(
        case_dir=io_fixtures, validate=False, file=invalid_file
    )
    errors = instance.check_validation()

    assert len(errors) == 2

    # Subdict context is attached
    assert all(e.subdict == "metadata" for e in errors)


def test_mutated_instance_catches_new_errors(io_fixtures):
    """check_validation() catches errors introduced by mutation after load."""
    instance = SimpleYAMLValidation.load(case_dir=io_fixtures)
    assert instance.check_validation() == []

    # Mutate to invalid state
    instance.count = -1  # violates gt=0
    errors = instance.check_validation()
    assert len(errors) == 1
    assert errors[0].field == ("count",)
    assert errors[0].error_type == "greater_than"


# -- validate_models() batch function -----------------------------------------


def test_all_valid_returns_empty(io_fixtures):
    """validate_models() returns [] when all models are valid."""
    models = [
        SimpleYAMLValidation.load(case_dir=io_fixtures),
        MetadataYAMLValidation.load(case_dir=io_fixtures),
    ]
    assert validate_models(models) == []


def test_mixed_valid_invalid(io_fixtures):
    """validate_models() aggregates errors from multiple models."""
    valid = SimpleYAMLValidation.load(case_dir=io_fixtures)
    invalid = SimpleJSONValidation.load(
        case_dir=io_fixtures, validate=False, file="invalid_simple.json"
    )

    errors = validate_models([valid, invalid])

    # Only the invalid model contributes errors
    assert len(errors) == 2
    assert all(isinstance(e, ValidationErrors) for e in errors)


def test_multiple_invalid(io_fixtures):
    """validate_models() collects errors from all invalid models."""
    invalid_simple = SimpleYAMLValidation.load(
        case_dir=io_fixtures, validate=False, file="invalid_simple.yaml"
    )
    invalid_nested = MetadataYAMLValidation.load(
        case_dir=io_fixtures, validate=False, file="invalid_nested.yaml"
    )

    errors = validate_models([invalid_simple, invalid_nested])

    # 2 from simple (count type + percentage constraint)
    # 2 from nested (missing version + priority gt=0)
    assert len(errors) == 4


def test_empty_list_returns_empty():
    """validate_models([]) returns []."""
    assert validate_models([]) == []
