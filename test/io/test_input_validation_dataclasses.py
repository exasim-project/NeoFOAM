# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

from pydantic import BaseModel
from foamadapter.io.input_validation import ModelInputDefinition, ValidationErrors


def test_file_spec_instance() -> None:
    class DummyModel(BaseModel):  # type: ignore[misc]
        value: int

    fs = ModelInputDefinition(
        baseModel=DummyModel, relative_path="system/controlDict", required=True
    )
    assert fs.relative_path == "system/controlDict"
    assert fs.baseModel == DummyModel
    assert fs.required is True
    assert fs.encoding == "utf-8"
    assert fs.description == ""


def test_validation_errors_instance() -> None:
    ve = ValidationErrors(
        field="velocity",
        error_type="TypeError",
        message="Expected float but got str",
        file_name="0/U",
        input_value="not_a_float",
    )
    assert ve.field == "velocity"
    assert ve.error_type == "TypeError"
    assert ve.message == "Expected float but got str"
    assert ve.file_name == "0/U"
    assert ve.input_value == "not_a_float"
