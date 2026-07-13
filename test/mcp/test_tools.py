# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

import shutil
from pathlib import Path
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
TUBE_BANK = Path(__file__).parent / "cases" / "tube_bank"


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
    assert (
        "properties" in block.step_schema and "tool" in block.step_schema["properties"]
    )
    # blockMesh reads system/blockMeshDict, written by BlockMeshDictConfig
    assert block.dict_file == "system/blockMeshDict"
    assert block.config == "BlockMeshDictConfig"
    assert by_name["snappyHexMesh"].config == "SnappyHexMeshDictConfig"
    # checkMesh reads no dict
    assert (
        by_name["checkMesh"].dict_file is None and by_name["checkMesh"].config is None
    )


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

    manifest = TUBE_BANK / "manifest.json"
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
    """Stage the tube-bank mesh dicts + a 0/U whose frontBack BC is ``frontback_bc``.

    The blockMeshDict / snappyHexMeshDict are checked-in fixtures (a constraint
    ``frontBack`` patch is what validate_case cross-checks against 0/U); only the U
    field varies per test, so the validator sees a real staged case.
    """
    from neofoam.workflow.patch_set import PatchSet
    from neofoam.framework.solver.configurations import configurations
    from neofoam.io import write_configs
    from neofoam.solver.incompressibleFluid.incompressibleFluid import (
        incompressibleFluid,
    )

    ps = PatchSet.load(TUBE_BANK / "manifest.json")
    system = tmp_path / "system"
    system.mkdir(parents=True, exist_ok=True)
    for name in ("blockMeshDict", "snappyHexMeshDict"):
        shutil.copy(TUBE_BANK / "system" / name, system / name)
    u = configurations(incompressibleFluid)["UFieldConfig"](
        boundaryField={
            p.name: {
                "type": frontback_bc if p.role.value == "empty" else "zeroGradient"
            }
            for p in ps.patches
        }
    )
    write_configs([u], case_dir=tmp_path)


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
    assert not any(
        f.file == "constant/g"
        for f in tools.validate_case(solver, str(tmp_path)).findings
    )


def test_validate_case_buoyant_p_needs_no_pfinal(solver: Any, tmp_path: Path) -> None:
    """In a Boussinesq case the vestigial 'p' solver needs no 'pFinal' (p_rgh is solved).

    The buoyantBoussinesq tutorials ship U/UFinal + p_rgh/p_rghFinal + a plain 'p'
    solver with NO pFinal — the validator must not demand one; a genuinely missing
    Final (here 'U' without 'UFinal') is still flagged.
    """
    _write(tmp_path / "constant" / "transportProperties", "beta 3e-3;\nTRef 300;\n")
    _write(
        tmp_path / "system" / "fvSolution",
        "solvers{ p{ solver GAMG; smoother GaussSeidel; }\n"
        "  p_rgh{ solver GAMG; smoother GaussSeidel; }\n"
        "  p_rghFinal{ solver GAMG; smoother GaussSeidel; }\n"
        "  U{ solver smoothSolver; } }\n",
    )
    findings = tools.validate_case(solver, str(tmp_path)).findings
    msgs = [f.message for f in findings]
    assert not any("pFinal" in m for m in msgs)  # p is vestigial → no pFinal needed
    assert any("UFinal" in m for m in msgs)  # a real missing Final is still flagged


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


def test_validate_case_errors_when_a_boundary_type_cannot_be_read(
    solver: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unreadable boundary type is an error, never a silently skipped check.

    Reproduces the false-success shape without depending on the backend raising:
    the reader reports the leaf as Unreadable, and validate_case must surface it.
    """
    from neofoam.framework.validation import checks as checks_mod

    def fake_read_section(path: Path, section: str) -> Any:
        if section == "boundaryField" and path.name == "U":
            return {
                "frontBack": {
                    "type": checks_mod.Unreadable(reason="not a single string")
                }
            }
        return {}

    monkeypatch.setattr(checks_mod, "read_section", fake_read_section)
    report = tools.validate_case(solver, str(tmp_path))

    assert not report.ok
    hit = next(
        f for f in report.findings if f.file == "0/U" and "frontBack" in f.message
    )
    assert hit.level == "error" and "could not be read" in hit.message


def test_validate_case_reads_boundary_through_dictread_without_truncation(
    solver: Any, tmp_path: Path
) -> None:
    """A field carrying a vector `value` leaf still gets its constraint mismatch flagged.

    Guards the extraction: a non-scalar sibling leaf must not truncate the
    boundaryField and hide the empty-vs-symmetry error.
    """
    _stage_mesh_dicts_with_u(tmp_path, frontback_bc="empty")  # mesh patch is symmetry
    _write(
        tmp_path / "0" / "U",
        "dimensions [0 1 -1 0 0 0 0];\ninternalField uniform (0 0 0);\n"
        "boundaryField{ frontBack{ type empty; }\n"
        "  inlet{ type fixedValue; value uniform (1 0 0); } }\n",
    )
    report = tools.validate_case(solver, str(tmp_path))
    assert any(
        f.file == "0/U" and "frontBack" in f.message and "symmetry" in f.message
        for f in report.findings
    )


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


def test_save_case_rejects_an_escaping_target(solver: Any, tmp_path: Path) -> None:
    from neofoam.tooling import CaseAccessError, Workspace

    ws = Workspace.at(tmp_path)
    with pytest.raises(CaseAccessError):
        tools.save_case(solver, {}, "../escape", workspace=ws)


@pytest.mark.parametrize(
    "verb", ["read_case", "load_case", "validate_case", "case_patches"]
)
def test_read_verbs_reject_an_escaping_case_dir(
    solver: Any, tmp_path: Path, verb: str
) -> None:
    from neofoam.tooling import CaseAccessError, Workspace

    ws = Workspace.at(tmp_path)
    fn = getattr(tools, verb)  # test-only reflection over the tool surface
    args = (solver, "../escape") if verb != "case_patches" else ("../escape",)
    with pytest.raises(CaseAccessError):
        fn(*args, workspace=ws)
