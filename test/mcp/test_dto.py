# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

from typing import Any

from neofoam.mcp.dto import (
    ConfigInfoDTO,
    ConfigSchemaDTO,
    ModelEntryDTO,
    SaveResultDTO,
    ToggleModelDTO,
)


class FooConfig:
    """A stand-in config class — its ``__name__`` is what the DTO must capture."""


class _FakeEntry:
    name = "Pimple"
    label = "Pimple"
    required = True
    dicts = [FooConfig]
    fields: list[Any] = []


def test_model_entry_dto_maps_classes_to_names() -> None:
    dto = ModelEntryDTO.from_entry(_FakeEntry())
    assert dto.dicts == ["FooConfig"]
    assert dto.fields == []
    text = dto.model_dump_json()
    assert "FooConfig" in text
    assert "class" not in text and "ModelMetaclass" not in text


def test_toggle_model_dto_maps_classes_to_names() -> None:
    dto = ToggleModelDTO.from_toggle(_FakeEntry())
    assert dto.dicts == ["FooConfig"]
    assert ModelEntryDTO.model_validate_json(
        ModelEntryDTO.from_entry(_FakeEntry()).model_dump_json()
    )


def test_config_info_and_schema_round_trip() -> None:
    info = ConfigInfoDTO(name="foo", cls_name="FooConfig", file="system/foo")
    assert (
        ConfigInfoDTO.model_validate_json(info.model_dump_json()).file == "system/foo"
    )
    schema = ConfigSchemaDTO(
        name="FooConfig", json_schema={"a": 1}, ui_schema={}, defaults={"x": 2}
    )
    assert schema.model_dump_json()
    save = SaveResultDTO(target_dir="/t", written=["system/foo"], case_spec={})
    assert SaveResultDTO.model_validate_json(save.model_dump_json()).written == [
        "system/foo"
    ]
