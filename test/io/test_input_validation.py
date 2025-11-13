import pytest
import os
from pathlib import Path
from pybFoam.io.model_base import IOModelBase
from pydantic import BaseModel, Field, ValidationError
from foamadapter.io.case_inputs import FileSpec, ValidationErrors, Registry

@pytest.fixture
def run_from_parent_directory():
    cwd = Path.cwd()
    parent_dir = Path(__file__).parent
    os.chdir(parent_dir)
    yield
    os.chdir(cwd)

def test_validate_case(run_from_parent_directory):
    registry = Registry()

    # Define a simple Pydantic model for testing
    class CorrectModel(IOModelBase):
        value: int
        too_high_value: int = Field(..., le=100)

    yaml_file = FileSpec(relative_path="input_files/simple_model.yaml", required=True)
    assert Path(yaml_file.relative_path).exists()

    registry["simple"] = (CorrectModel, yaml_file)
    success, errors = registry.validate_case(case_dir=".")
    assert success is True
    assert len(errors) == 0

def test_validation_error(run_from_parent_directory):
    registry = Registry()

    # Define a simple Pydantic model for testing
    class IncorrectModel(IOModelBase):
        missing_value: int
        too_high_value: int = Field(..., le=10)

    yaml_file = FileSpec(relative_path="input_files/simple_model.yaml", required=True)
    assert Path(yaml_file.relative_path).exists()

    registry["simple"] = (IncorrectModel, yaml_file)

    success, errors = registry.validate_case(case_dir=".")
    assert not success
    assert len(errors) == 2

    first_error = errors[0]
    
    assert first_error.error_type == "missing"
    assert first_error.file_name == "input_files/simple_model.yaml"
    assert first_error.input_value == {'too_high_value': 11}
    assert first_error.field == ("missing_value",)
    assert first_error.message == "Field required"

    second_error = errors[1]
    assert second_error.error_type == "less_than_equal"
    assert second_error.file_name == "input_files/simple_model.yaml"
    assert second_error.input_value == 11
    assert second_error.field == ("too_high_value",)
    assert second_error.message == 'Input should be less than or equal to 10'
 


    