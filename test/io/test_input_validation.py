import os
from pathlib import Path

import pytest
from pybFoam.io.model_base import IOModelBase
from pydantic import BaseModel, Field

from foamadapter.io.input_validation import ModelInputDefinition, ModelInputCollection


@pytest.fixture
def run_from_parent_directory():
    cwd = Path.cwd()
    parent_dir = Path(__file__).parent
    os.chdir(parent_dir)
    yield
    os.chdir(cwd)


def test_validate_case(run_from_parent_directory):
    registry = ModelInputCollection()

    # Define a simple Pydantic model for testing
    class CorrectModel(IOModelBase):
        value: int
        too_high_value: int = Field(..., le=100)

    correct_model = ModelInputDefinition(
        baseModel=CorrectModel, relative_path="input_files/simple_model.yaml", required=True
    )
    assert Path(correct_model.relative_path).exists()

    registry.add(correct_model)
    success, errors = registry.validate_case(case_dir=".")
    assert success is True
    assert len(errors) == 0


def test_validation_error(run_from_parent_directory):
    registry = ModelInputCollection()

    # Define a simple Pydantic model for testing
    class IncorrectModel(IOModelBase):
        missing_value: int
        too_high_value: int = Field(..., le=10)

    incorrect_model = ModelInputDefinition(
        baseModel=IncorrectModel, relative_path="input_files/simple_model.yaml", required=True
    )
    assert Path(incorrect_model.relative_path).exists()

    registry.add(incorrect_model)

    success, errors = registry.validate_case(case_dir=".")
    assert not success
    assert len(errors) == 2

    first_error = errors[0]

    assert first_error.error_type == "missing"
    assert first_error.file_name == "input_files/simple_model.yaml"
    # assert first_error.input_value == {'too_high_value': 11}
    assert first_error.field == ("missing_value",)
    assert first_error.message == "Field required"

    second_error = errors[1]
    assert second_error.error_type == "less_than_equal"
    assert second_error.file_name == "input_files/simple_model.yaml"
    assert second_error.input_value == 11
    assert second_error.field == ("too_high_value",)
    assert second_error.message == "Input should be less than or equal to 10"


def test_registry_find_remove():
    registry = ModelInputCollection()

    class DummyModelA(BaseModel):
        pass

    class DummyModelB(BaseModel):
        pass

    input_def_a = ModelInputDefinition(
        baseModel=DummyModelA, relative_path="path/to/a.yaml", required=True
    )
    input_def_b = ModelInputDefinition(
        baseModel=DummyModelB, relative_path="path/to/b.yaml", required=False
    )

    registry.add(input_def_a)
    registry.add(input_def_b)

    # Test find by index
    assert registry.find(0) == input_def_a
    assert registry.find(1) == input_def_b

    # Test find by path
    assert registry.find("path/to/a.yaml") == input_def_a
    assert registry.find(Path("path/to/b.yaml")) == input_def_b

    # Test find by model type
    assert registry.find(DummyModelA) == input_def_a
    assert registry.find(DummyModelB) == input_def_b

    # Test remove
    registry.remove(0)
    assert len(registry._inputs) == 1
    assert registry.find(0) == input_def_b
