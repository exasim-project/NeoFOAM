# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""Tests for OpenFOAM IO decorator patterns via pybFoam."""

import pytest

from pydantic import Field, ValidationError
from neofoam.io import BaseConfig, IOStrategy, OPENFOAM


@IOStrategy(OPENFOAM("simple.of"))
class SimpleOpenFOAMConfig(BaseConfig):
    identifier: str
    count: int = Field(gt=0)
    active: bool
    percentage: float = Field(ge=0, le=100)


@IOStrategy(OPENFOAM("nested.of", subdict="metadata"))
class MetadataOpenFOAMConfig(BaseConfig):
    name: str
    version: float
    priority: int = Field(gt=0)


@IOStrategy(OPENFOAM("nested.of", subdict="processing.stage1"))
class Stage1OpenFOAMConfig(BaseConfig):
    algorithm: str
    threshold: float = Field(gt=0)
    maxIterations: int = Field(gt=0, le=1000)
    batchSize: int = Field(gt=0)


@IOStrategy(OPENFOAM("shared.of", subdict="config.service_a"))
class ServiceAOpenFOAMConfig(BaseConfig):
    timeout: int = Field(gt=0)
    maxConnections: int = Field(gt=0, le=100)


@IOStrategy(OPENFOAM("shared.of", subdict="config.service_b"))
class ServiceBOpenFOAMConfig(BaseConfig):
    endpoint: str
    port: int = Field(gt=0, le=65535)
    poolSize: int = Field(gt=0)


@IOStrategy(OPENFOAM("shared.of", subdict="config.service_c"))
class ServiceCOpenFOAMConfig(BaseConfig):
    enabled: bool
    bufferSize: int = Field(gt=0)


def test_load_simple(io_fixtures):
    loaded = SimpleOpenFOAMConfig.load(case_dir=io_fixtures)

    assert loaded.identifier == "test_config"
    assert loaded.count == 42
    assert loaded.active is True
    assert loaded.percentage == 75.5


def test_write_simple_roundtrip(tmp_path):
    config = SimpleOpenFOAMConfig(
        identifier="demo", count=100, active=False, percentage=50.0
    )

    output_file = tmp_path / "simple_written.of"
    config.save(case_dir=output_file)

    loaded = SimpleOpenFOAMConfig.load(case_dir=output_file)
    assert loaded.model_dump() == config.model_dump()


def test_load_nested(io_fixtures):
    metadata = MetadataOpenFOAMConfig.load(case_dir=io_fixtures)
    stage1 = Stage1OpenFOAMConfig.load(case_dir=io_fixtures)

    assert metadata.name == "TestApp"
    assert metadata.version == 1.0
    assert metadata.priority == 5

    assert stage1.algorithm == "fast"
    assert stage1.threshold == 0.001
    assert stage1.maxIterations == 100
    assert stage1.batchSize == 32


def test_validation_error_wrong_type(io_fixtures):
    loaded_data = SimpleOpenFOAMConfig.load(
        case_dir=io_fixtures, validate=False, file="invalid_simple.of"
    )
    assert loaded_data.identifier == "test_config"
    assert loaded_data.count == "not_an_integer"
    assert loaded_data.active is True
    assert loaded_data.percentage == 150

    with pytest.raises(ValidationError) as exc_info:
        SimpleOpenFOAMConfig.load(
            case_dir=io_fixtures, validate=True, file="invalid_simple.of"
        )

    errors = exc_info.value.errors()
    assert len(errors) == 2

    int_errors = [e for e in errors if e["loc"] == ("count",)]
    assert len(int_errors) == 1
    assert int_errors[0]["type"] == "int_parsing"

    le_errors = [e for e in errors if e["loc"] == ("percentage",)]
    assert len(le_errors) == 1
    assert le_errors[0]["type"] == "less_than_equal"
    assert le_errors[0]["ctx"]["le"] == 100


def test_validation_error_missing_field_in_subdict(io_fixtures):
    loaded_data = MetadataOpenFOAMConfig.load(
        case_dir=io_fixtures, validate=False, file="invalid_nested.of"
    )
    assert loaded_data.name == "TestApp"
    assert loaded_data.priority == 0

    with pytest.raises(ValidationError) as exc_info:
        MetadataOpenFOAMConfig.load(
            case_dir=io_fixtures, validate=True, file="invalid_nested.of"
        )

    errors = exc_info.value.errors()
    assert len(errors) == 2

    missing_errors = [e for e in errors if e["type"] == "missing"]
    assert len(missing_errors) == 1
    assert missing_errors[0]["loc"] == ("version",)

    gt_errors = [e for e in errors if e["type"] == "greater_than"]
    assert len(gt_errors) == 1
    assert gt_errors[0]["loc"] == ("priority",)
    assert gt_errors[0]["ctx"]["gt"] == 0


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        SimpleOpenFOAMConfig.load(case_dir=tmp_path, file="nonexistent.of")


def test_load_missing_subdict_raises(io_fixtures):
    with pytest.raises(KeyError, match="not found"):
        MetadataOpenFOAMConfig.load(case_dir=io_fixtures, file="simple.of")


def test_write_preserves_other_subdicts(temp_fixture_copy):
    test_file = temp_fixture_copy("shared.of")
    test_dir = test_file.parent

    original_b = ServiceBOpenFOAMConfig.load(case_dir=test_dir)
    original_c = ServiceCOpenFOAMConfig.load(case_dir=test_dir)

    service_a = ServiceAOpenFOAMConfig.load(case_dir=test_dir)
    service_a.timeout = 60
    service_a.save(case_dir=test_dir)

    updated_a = ServiceAOpenFOAMConfig.load(case_dir=test_dir)
    updated_b = ServiceBOpenFOAMConfig.load(case_dir=test_dir)
    updated_c = ServiceCOpenFOAMConfig.load(case_dir=test_dir)

    assert updated_a.timeout == 60
    assert updated_a.maxConnections == 10
    assert updated_b.model_dump() == original_b.model_dump()
    assert updated_c.model_dump() == original_c.model_dump()
