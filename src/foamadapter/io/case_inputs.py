from dataclasses import dataclass
from typing import Annotated, Any, Dict, Tuple, Type
from pydantic import BaseModel, Field, create_model, ValidationError
from pathlib import Path

# TODO split into class
# Change the architecture Create a New Class that holds the type and the FileSpec
# and provide a file loading strategy there
@dataclass(frozen=True)
class FileSpec:
    relative_path: str
    encoding: str = "utf-8"
    required: bool = True
    description: str = ""


@dataclass(frozen=True)
class ValidationErrors:
    field: Any
    error_type: str
    message: str
    file_name: str
    input_value: Any = None


class Registry(Dict[str, Tuple[Type[BaseModel], FileSpec]]):
    """Registry that holds model classes and their file specifications."""

    def build_case_inputs_class(self, name: str = "CaseInputs") -> Type[BaseModel]:
        """
        Build a Pydantic model class from the registry that can validate all inputs.

        Args:
            name: Name for the generated class

        Returns:
            A Pydantic model class that can hold all the input files
        """
        fields = {}
        for field_name, (model_type, spec) in self.items():
            ann = Annotated[model_type, spec]  # attach FileSpec via Annotated
            fields[field_name] = (ann, ...)  # "..." means required

        CaseInputs = create_model(name, **fields)
        return CaseInputs

    def read_case_inputs(self, case_dir: str = ".", name: str = "CaseInputs"):
        """
        Read all input files from a case directory.

        Args:
            case_dir: Path to the OpenFOAM case directory
            name: Name for the generated class

        Returns:
            Instance of the case inputs class with all files loaded
        """
        case_path = Path(case_dir)
        inputs_class = self.build_case_inputs_class(name)
        loaded_data = {}

        for key, (model_class, file_spec) in self.items():
            file_path = case_path / file_spec.relative_path

            try:
                if file_path.exists():
                    loaded_model = model_class.from_file(str(file_path))
                    loaded_data[key] = loaded_model
                else:
                    if file_spec.required:
                        print(f"Required file not found: {file_path}")
                        raise FileNotFoundError(f"Required file not found: {file_path}")
                    else:
                        print(f"Optional file not found: {file_path}")
            except Exception as e:
                print(f"Error loading {key} from {file_path}: {e}")
                if file_spec.required:
                    raise

        return inputs_class(**loaded_data)

    def validate_case(self, case_dir: str = ".") -> Tuple[bool, list[ValidationErrors]]:
        """
        Validate that all required files exist and are valid for the given case.

        Args:
            case_dir: Path to the OpenFOAM case directory

        Returns:
            Tuple of (is_valid: bool, errors: List[ValidationErrors])
        """
        case_path = Path(case_dir)
        validation_errors: list[ValidationErrors] = []

        for key, (model_class, file_spec) in self.items():
            file_path = case_path / file_spec.relative_path

            if not file_path.exists():
                if file_spec.required:
                    validation_errors.append(
                        ValidationErrors(
                            field=None,
                            message="File not found",
                            file_name=str(file_path),
                            error_type="FileNotFound",
                        )
                    )
                continue

            try:
                # Try to load and validate the file
                model_class.from_file(str(file_path))
            except ValidationError as errors:
                for err in errors.errors(include_url=False):
                    # Collect validation errors
                    validation_errors.append(
                        ValidationErrors(
                            field=err.get("loc", [None]),
                            message=err.get("msg", "Validation error"),
                            file_name=str(file_path),
                            input_value=err.get("input", None),
                            error_type=err.get("type", "UnknownError"),
                        )
                    )

        return len(validation_errors) == 0, validation_errors
