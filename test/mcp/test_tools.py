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

pytest.importorskip("pybFoam")  # resolving the solver spec imports pybFoam

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
        "tool_catalog",
        "list_configs",
        "config_schema",
        "case_patches",
        "validate_case",
        "read_case",
        "load_case",
        "save_case",
        "fill_case",
    )


def test_list_solvers_lists_incompressible_fluid() -> None:
    assert "incompressibleFluid" in tools.list_solvers()


def test_tool_catalog_lists_tools_with_step_schema_and_config(solver: Any) -> None:
    """Each preprocessing tool is catalogued with its entry schema + the config it reads."""
    catalog = tools.tool_catalog(solver)
    by_name = {t.name: t for t in catalog}
    assert {"blockMesh", "snappyHexMesh", "checkMesh"} <= set(by_name)
    # the preprocess.yaml entry schema (its ``tool`` key + options)
    block = by_name["blockMesh"]
    assert "properties" in block.step_schema and "tool" in block.step_schema["properties"]
    # blockMesh reads system/blockMeshDict, written by BlockMeshDictConfig
    assert block.dict_file == "system/blockMeshDict"
    assert block.config == "BlockMeshDictConfig"
    assert by_name["snappyHexMesh"].config == "SnappyHexMeshDictConfig"
    # checkMesh reads no dict
    assert by_name["checkMesh"].dict_file is None and by_name["checkMesh"].config is None


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


def test_config_schema_accepts_snake_case_name(solver: Any) -> None:
    """The snake_case name list_configs advertises round-trips into config_schema."""
    dto = tools.config_schema(solver, "control_dict_config")
    assert dto.json_schema and "properties" in dto.json_schema


def test_case_patches_reads_staged_manifest(tmp_path: Path) -> None:
    """case_patches returns the boundary patches (name + role) from the manifest."""
    from neofoam.workflow.patch_set import PatchSet

    manifest = REPO_ROOT / "test" / "workflow" / "cases" / "tube_bank_manifest.json"
    PatchSet.load(manifest).save(tmp_path / "manifest.json")
    patches = tools.case_patches(str(tmp_path))
    assert {p.name: p.role for p in patches} == {
        "inlet": "inlet",
        "outlet": "outlet",
        "walls": "wall",
        "tubes": "wall",
        "frontBack": "empty",
    }


def test_case_patches_missing_manifest_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        tools.case_patches(str(tmp_path))  # dir exists but has no manifest.json


def _stage_mesh_dicts_with_u(tmp_path: Path, frontback_bc: str) -> Any:
    """Write the tube-bank mesh dicts + a 0/U whose frontBack BC is ``frontback_bc``."""
    from neofoam.workflow.mesh_inputs import block_mesh_dict, snappy_dict
    from neofoam.workflow.patch_set import PatchSet
    from neofoam.framework.solver.configurations import configurations
    from neofoam.io import write_configs
    from neofoam.solver.incompressibleFluid.incompressibleFluid import incompressibleFluid

    ps = PatchSet.load(REPO_ROOT / "test" / "workflow" / "cases" / "tube_bank_manifest.json")
    u = configurations(incompressibleFluid)["UFieldConfig"](
        boundaryField={
            p.name: {"type": frontback_bc if p.role.value == "empty" else "zeroGradient"}
            for p in ps.patches
        }
    )
    write_configs([block_mesh_dict(ps), snappy_dict(ps), u], case_dir=tmp_path)


def test_validate_case_flags_constraint_patch_mismatch(
    solver: Any, tmp_path: Path
) -> None:
    """A BC type that doesn't match a constraint mesh patch is an error with a fix."""
    _stage_mesh_dicts_with_u(tmp_path, frontback_bc="empty")  # mesh patch is symmetry
    report = tools.validate_case(solver, str(tmp_path))
    assert not report.ok
    hit = next(
        f
        for f in report.findings
        if f.file == "0/U" and "frontBack" in f.message and "symmetry" in f.message
    )
    assert hit.fix and "symmetry" in hit.fix


def test_validate_case_accepts_matching_constraint_patch(
    solver: Any, tmp_path: Path
) -> None:
    """With the correct symmetry BC, there is no patch-mismatch finding for frontBack."""
    _stage_mesh_dicts_with_u(tmp_path, frontback_bc="symmetry")
    report = tools.validate_case(solver, str(tmp_path))
    assert not any("frontBack" in f.message for f in report.findings)


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "FoamFile{ version 2.0; format ascii; class dictionary; object x; }\n" + body
    )


def test_validate_case_requires_constant_g_when_boussinesq(
    solver: Any, tmp_path: Path
) -> None:
    """Boussinesq (beta+TRef in transportProperties) without constant/g is an error."""
    _write(tmp_path / "constant" / "transportProperties", "beta 3e-3;\nTRef 300;\n")
    report = tools.validate_case(solver, str(tmp_path))
    hit = next(f for f in report.findings if f.file == "constant/g")
    assert hit.level == "error" and hit.fix and "value (0 -9.81 0)" in hit.fix
    # authoring the config clears the finding
    from neofoam.solver.incompressibleFluid.models.boussinesq import GravityConfig

    from neofoam.io import write_configs

    write_configs([GravityConfig()], case_dir=tmp_path)
    assert not any(f.file == "constant/g" for f in tools.validate_case(solver, str(tmp_path)).findings)


def test_validate_case_flags_wall_function_in_laminar_case(
    solver: Any, tmp_path: Path
) -> None:
    """A ``*WallFunction`` BC is invalid when turbulence is laminar; fix → calculated."""
    _write(tmp_path / "constant" / "turbulenceProperties", "simulationType laminar;\n")
    _write(
        tmp_path / "0" / "alphat",
        "boundaryField{ walls{ type compressible::alphatWallFunction; value uniform 0; }\n"
        "inlet{ type calculated; value uniform 0; } }\n",
    )
    report = tools.validate_case(solver, str(tmp_path))
    hit = next(f for f in report.findings if f.file == "0/alphat")
    assert "walls" in hit.message and "laminar" in hit.message
    assert hit.fix and "calculated" in hit.fix
    # 'calculated' on inlet is fine — only the wall-function patch is flagged
    assert sum(1 for f in report.findings if f.file == "0/alphat") == 1


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
