# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from neofoam.mcp import tools
from neofoam.mcp.dto import (
    CaseSpecDTO,
    CaseTextDTO,
    ConfigInfoDTO,
    ConfigSchemaDTO,
    ModelEntryDTO,
)
from neofoam.mcp.tools import ALL_TOOL_NAMES


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_CASE = REPO_ROOT / "test" / "solver" / "incompressibleFluid" / "val_pitzDaily"


@pytest.fixture
def solver() -> Any:
    from neofoam.mcp.registry import resolve_solver

    return resolve_solver("incompressibleFluid")


def test_all_tool_names_match_the_spec_literal_list() -> None:
    assert ALL_TOOL_NAMES == (
        "list_solvers",
        "model_catalog",
        "list_configs",
        "config_schema",
        "read_case",
        "load_case",
        "save_case",
        "fill_case",
    )


def test_list_solvers_lists_incompressible_fluid() -> None:
    assert "incompressibleFluid" in tools.list_solvers()


def test_model_catalog_is_case_free_with_expected_names(solver: Any) -> None:
    entries = tools.model_catalog(solver)
    assert all(isinstance(e, ModelEntryDTO) for e in entries)
    names = {e.name for e in entries}
    assert {"courant", "maxDeltaT", "boussinesq"} <= names
    required = {e.name for e in entries if e.required}
    assert {"Pimple", "Newtonian", "laminar"} <= required
    # Serialized owned configs are class-name strings, no class repr.
    text = ModelEntryDTO(
        name="x",
        label="x",
        required=True,
        dicts=entries[0].dicts,
        fields=entries[0].fields,
    ).model_dump_json()
    assert "ModelMetaclass" not in text


def test_list_configs_includes_control_dict_with_file(solver: Any) -> None:
    configs = tools.list_configs(solver)
    assert all(isinstance(c, ConfigInfoDTO) for c in configs)
    cd = next(c for c in configs if c.cls_name == "ControlDictConfig")
    assert cd.file == "system/controlDict"


def test_config_schema_has_schema_ui_and_defaults(solver: Any) -> None:
    dto = tools.config_schema(solver, "ControlDictConfig")
    assert isinstance(dto, ConfigSchemaDTO)
    assert dto.json_schema and "properties" in dto.json_schema
    assert dto.defaults["application"] == "pimpleFoam"
    assert isinstance(dto.ui_schema, dict)


def test_config_schema_unknown_name_errors(solver: Any) -> None:
    with pytest.raises(ValueError):
        tools.config_schema(solver, "NoSuchConfig")


def test_read_case_returns_file_texts(solver: Any) -> None:
    dto = tools.read_case(solver, str(SOURCE_CASE))
    assert isinstance(dto, CaseTextDTO)
    assert "system/controlDict" in dto.files
    assert dto.files["system/controlDict"].strip()


def test_load_case_returns_case_spec_dump(solver: Any) -> None:
    dto = tools.load_case(solver, str(SOURCE_CASE))
    assert isinstance(dto, CaseSpecDTO)
    assert dto.values["control_dict_config"] is not None


def test_read_case_missing_dir_raises_naming_path(solver: Any, tmp_path: Path) -> None:
    missing = tmp_path / "no_such_case"
    with pytest.raises(ValueError, match=str(missing)):
        tools.read_case(solver, str(missing))


def test_load_case_missing_dir_raises_naming_path(solver: Any, tmp_path: Path) -> None:
    missing = tmp_path / "no_such_case"
    with pytest.raises(ValueError, match=str(missing)):
        tools.load_case(solver, str(missing))


def test_save_case_validates_then_writes(solver: Any, tmp_path: Path) -> None:
    from neofoam.agent.case_fill import load_case_from_disk

    spec_dump = load_case_from_disk(SOURCE_CASE, solver=solver).model_dump()

    result = tools.save_case(solver, spec_dump, str(tmp_path))

    assert result.written, "no files reported as written"
    for rel in ("system/controlDict", "constant/transportProperties"):
        assert (tmp_path / rel).exists()
    assert result.case_spec["control_dict_config"] is not None


def test_save_case_rejects_malformed_without_writing(
    solver: Any, tmp_path: Path
) -> None:
    target = tmp_path / "out"
    target.mkdir()
    sentinel = target / "keepme.txt"
    sentinel.write_text("pre-existing")

    with pytest.raises(ValueError):
        tools.save_case(solver, {"control_dict_config": "not a dict"}, str(target))

    # No partial write: the sentinel is untouched and nothing new was created.
    assert sentinel.read_text() == "pre-existing"
    assert list(target.iterdir()) == [sentinel]


class _StubAgent:
    """Network-free stand-in: run_sync mirrors pydantic-ai (calls asyncio.run)."""

    def __init__(self, output: object) -> None:
        self._output = output
        self.received_prompt: str | None = None

    def run_sync(self, prompt: str) -> SimpleNamespace:
        async def _noop() -> None:
            return None

        asyncio.run(_noop())  # raises if a loop is already running in this thread
        self.received_prompt = prompt
        return SimpleNamespace(output=self._output)


def test_fill_case_offloads_blocking_agent_call(solver: Any, tmp_path: Path) -> None:
    from neofoam.agent.case_fill import load_case_from_disk

    prebuilt = load_case_from_disk(SOURCE_CASE, solver=solver)
    target = tmp_path / "filled"

    async def _invoke() -> Any:
        return await tools.fill_case(
            solver,
            str(SOURCE_CASE),
            str(target),
            agent_factory=lambda *, solver, model_name: _StubAgent(prebuilt),
        )

    result = asyncio.run(_invoke())  # no RuntimeError about the event loop

    assert result.written
    assert (target / "system" / "controlDict").exists()


def test_fill_case_includes_optional_prompt_in_agent_call(
    solver: Any, tmp_path: Path
) -> None:
    from neofoam.agent.case_fill import load_case_from_disk

    prebuilt = load_case_from_disk(SOURCE_CASE, solver=solver)
    stub = _StubAgent(prebuilt)
    target = tmp_path / "filled"

    async def _invoke() -> Any:
        return await tools.fill_case(
            solver,
            str(SOURCE_CASE),
            str(target),
            "make it laminar",
            agent_factory=lambda *, solver, model_name: stub,
        )

    result = asyncio.run(_invoke())

    assert result.written
    assert stub.received_prompt is not None
    assert "make it laminar" in stub.received_prompt
