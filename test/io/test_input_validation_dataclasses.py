from foamadapter.io.case_inputs import FileSpec, ValidationErrors


def test_file_spec_instance():
    fs = FileSpec(relative_path="system/controlDict", required=True)
    assert fs.relative_path == "system/controlDict"
    assert fs.required is True
    assert fs.encoding == "utf-8"
    assert fs.description == ""

def test_validation_errors_instance():
    ve = ValidationErrors(
        field="velocity",
        error_type="TypeError",
        message="Expected float but got str",
        file_name="0/U",
        input_value="not_a_float"
    )
    assert ve.field == "velocity"
    assert ve.error_type == "TypeError"
    assert ve.message == "Expected float but got str"
    assert ve.file_name == "0/U"
    assert ve.input_value == "not_a_float"