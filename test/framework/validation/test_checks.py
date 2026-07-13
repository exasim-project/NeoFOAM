# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

from pathlib import Path
from typing import Any

import pytest

from neofoam.framework.validation import CaseContext, validate
from neofoam.framework.validation import checks as checks_mod
from neofoam.framework.validation.checks import (
    _expand_group,
    check_boussinesq_gravity,
    check_constraint_patches,
    check_div_scheme,
    check_gamg_smoother,
    check_laminar_wall_functions,
    check_pimple_final,
)

CASES = Path(__file__).parent / "cases"


def _ctx(case: Path) -> CaseContext:
    return CaseContext(case=case, solver=object())


# -- grouped-solver-name expansion --------------------------------------------


def test_expand_group_plain_name_is_itself() -> None:
    assert _expand_group("U") == ["U"]


def test_expand_group_splits_a_grouped_key() -> None:
    assert _expand_group("(U|k|epsilon)") == ["U", "k", "epsilon"]


def test_expand_group_strips_surrounding_quotes() -> None:
    assert _expand_group('"(p|p_rgh)"') == ["p", "p_rgh"]


def test_pimple_final_accepts_grouped_base_with_individual_finals(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A runnable case: one grouped solver key plus per-field Final entries. The old
    # literal '(U|k|epsilon)Final' lookup false-failed this; matching by expanded
    # field accepts it.
    solvers = {
        "(U|k|epsilon)": {"solver": checks_mod.Value(text="smoothSolver")},
        "UFinal": {"solver": checks_mod.Value(text="smoothSolver")},
        "kFinal": {"solver": checks_mod.Value(text="smoothSolver")},
        "epsilonFinal": {"solver": checks_mod.Value(text="smoothSolver")},
        "p": {"solver": checks_mod.Value(text="GAMG")},
        "pFinal": {"solver": checks_mod.Value(text="GAMG")},
    }
    monkeypatch.setattr(checks_mod, "read_section", lambda path, section: solvers)
    monkeypatch.setattr(checks_mod, "is_boussinesq", lambda case: False)
    assert check_pimple_final(_ctx(tmp_path)) == []


def test_pimple_final_flags_a_grouped_field_missing_its_final(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    solvers = {
        "(U|k)": {"solver": checks_mod.Value(text="smoothSolver")},
        "UFinal": {"solver": checks_mod.Value(text="smoothSolver")},
    }
    monkeypatch.setattr(checks_mod, "read_section", lambda path, section: solvers)
    monkeypatch.setattr(checks_mod, "is_boussinesq", lambda case: False)
    findings = check_pimple_final(_ctx(tmp_path))
    assert [f.message for f in findings] == [
        "PIMPLE needs a 'kFinal' solver entry (missing)"
    ]


# -- couldn't-check ⇒ error at the three ported read sites ---------------------


def test_gamg_check_errors_when_solver_leaf_unreadable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        checks_mod,
        "read_section",
        lambda path, section: {
            "p": {"solver": checks_mod.Unreadable(reason="not a name")}
        },
    )
    findings = check_gamg_smoother(_ctx(tmp_path))
    assert findings and findings[0].level == "error"
    assert "could not be read" in findings[0].message


def test_laminar_check_errors_when_type_leaf_unreadable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(checks_mod, "turbulence_type", lambda case: "laminar")
    monkeypatch.setattr(
        checks_mod,
        "read_section",
        lambda path, section: {
            "walls": {"type": checks_mod.Unreadable(reason="not a name")}
        },
    )
    findings = check_laminar_wall_functions(_ctx(tmp_path))
    assert any(
        f.level == "error" and "could not be read" in f.message for f in findings
    )


def test_div_check_errors_when_scheme_leaf_unreadable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        checks_mod,
        "read_entry",
        lambda path, section, key: checks_mod.Unreadable(reason="bad"),
    )
    findings = check_div_scheme(_ctx(tmp_path))
    assert findings and findings[0].level == "error"


def test_div_check_absent_scheme_is_silent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Absence (None) is a genuine no-op — only a present-but-unreadable leaf escalates.
    monkeypatch.setattr(checks_mod, "read_entry", lambda path, section, key: None)
    assert check_div_scheme(_ctx(tmp_path)) == []


def test_constraint_check_errors_when_type_leaf_unreadable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        checks_mod,
        "read_section",
        lambda path, section: (
            {"frontBack": {"type": checks_mod.Unreadable(reason="bad")}}
            if path.name == "U"
            else {}
        ),
    )
    findings = check_constraint_patches(_ctx(tmp_path))
    assert any(
        f.level == "error"
        and "frontBack" in f.message
        and "could not be read" in f.message
        for f in findings
    )


# -- real-fixture end-to-end (proves the pybFoam grouped-key round-trip) -------


def test_validate_passes_a_grouped_runnable_fvsolution() -> None:
    pytest.importorskip("pybFoam")
    from neofoam.mcp.registry import resolve_solver

    solver: Any = resolve_solver("incompressibleFluid")
    report = validate(solver, CASES / "grouped_case")
    # No PIMPLE-final finding: the grouped base is satisfied by its per-field Finals.
    assert not any("Final" in f.message for f in report.findings)


# -- non-leaf helper escalation -----------------------------------------------


def test_mesh_patch_types_escalates_a_corrupt_blockmeshdict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A present blockMeshDict whose config load fails is Unreadable, not an empty set.
    import neofoam.tools.block_mesh as bm

    (tmp_path / "system").mkdir()
    (tmp_path / "system" / "blockMeshDict").write_text("garbage")

    def _boom(cls: Any, **kw: Any) -> Any:
        raise ValueError("blockMeshDict is corrupt")

    monkeypatch.setattr(bm.BlockMeshDictConfig, "load", classmethod(_boom))
    result = checks_mod.mesh_patch_types(tmp_path)
    assert isinstance(result, checks_mod.Unreadable)
    assert "blockMeshDict" in result.reason


def test_mesh_patch_types_escalates_a_corrupt_snappyhexmeshdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A snappyHexMeshDict that will not load surfaces as Unreadable, not a swallow."""
    from neofoam.tools import snappy_hex_mesh

    shm = tmp_path / "system" / "snappyHexMeshDict"
    shm.parent.mkdir(parents=True)
    shm.write_text(
        "FoamFile{ version 2.0; format ascii; class dictionary; "
        "object snappyHexMeshDict; }\n"
    )

    def _boom(cls: Any, **kw: Any) -> Any:
        raise RuntimeError("bad snappyHexMeshDict")

    monkeypatch.setattr(
        snappy_hex_mesh.SnappyHexMeshDictConfig, "load", classmethod(_boom)
    )
    result = checks_mod.mesh_patch_types(tmp_path)
    assert isinstance(result, checks_mod.Unreadable)
    assert "snappyHexMeshDict" in result.reason


def test_mesh_patch_types_absent_dicts_are_empty(tmp_path: Path) -> None:
    # Genuine absence stays an empty dict (not Unreadable) — absence != corruption.
    assert checks_mod.mesh_patch_types(tmp_path) == {}


def test_is_boussinesq_escalates_a_corrupt_transport(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "constant").mkdir()
    (tmp_path / "constant" / "transportProperties").write_text("x")
    monkeypatch.setattr(
        checks_mod, "read_keys", lambda path: checks_mod.Unreadable(reason="corrupt")
    )
    assert isinstance(checks_mod.is_boussinesq(tmp_path), checks_mod.Unreadable)


def test_is_boussinesq_absent_transport_is_false(tmp_path: Path) -> None:
    assert checks_mod.is_boussinesq(tmp_path) is False


def test_turbulence_type_escalates_a_corrupt_turbulence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        checks_mod,
        "read_toplevel",
        lambda path, key: checks_mod.Unreadable(reason="corrupt"),
    )
    assert isinstance(checks_mod.turbulence_type(tmp_path), checks_mod.Unreadable)


def test_turbulence_type_absent_is_none(tmp_path: Path) -> None:
    assert checks_mod.turbulence_type(tmp_path) is None


# -- dependent checks escalate the helper Unreadable (live path) ---------------


def test_constraint_check_errors_when_mesh_dict_is_corrupt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        checks_mod,
        "mesh_patch_types",
        lambda case: checks_mod.Unreadable(reason="system/blockMeshDict: corrupt"),
    )
    findings = check_constraint_patches(_ctx(tmp_path))
    assert [f.level for f in findings] == ["error"]
    assert "could not be read" in findings[0].message


def test_registry_ok_false_when_mesh_dict_is_corrupt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The live path the reviewer flagged: a corrupt blockMeshDict must make ok=False,
    # not pass silently. Run just the constraint check through a CheckRegistry.
    from neofoam.framework.validation import CheckRegistry

    monkeypatch.setattr(
        checks_mod,
        "mesh_patch_types",
        lambda case: checks_mod.Unreadable(reason="system/blockMeshDict: corrupt"),
    )
    reg = CheckRegistry()
    reg.add("constraint-patches", check_constraint_patches)
    assert reg.run(_ctx(tmp_path)).ok is False


def test_pimple_final_errors_when_transport_unreadable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        checks_mod,
        "read_section",
        lambda path, section: {"U": {"solver": checks_mod.Value(text="x")}},
    )
    monkeypatch.setattr(
        checks_mod,
        "is_boussinesq",
        lambda case: checks_mod.Unreadable(reason="corrupt"),
    )
    findings = check_pimple_final(_ctx(tmp_path))
    assert any(f.level == "error" and "Boussinesq" in f.message for f in findings)


def test_boussinesq_gravity_errors_when_transport_unreadable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        checks_mod,
        "is_boussinesq",
        lambda case: checks_mod.Unreadable(reason="corrupt"),
    )
    findings = check_boussinesq_gravity(_ctx(tmp_path))
    assert [f.level for f in findings] == ["error"]


def test_laminar_errors_when_turbulence_unreadable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        checks_mod,
        "turbulence_type",
        lambda case: checks_mod.Unreadable(reason="corrupt"),
    )
    findings = check_laminar_wall_functions(_ctx(tmp_path))
    assert [f.level for f in findings] == ["error"]


# -- GAMG / div positive findings from real fixtures --------------------------


def test_gamg_check_flags_a_gamg_solver_without_a_smoother() -> None:
    pytest.importorskip("pybFoam")
    findings = check_gamg_smoother(_ctx(CASES / "gamg_no_smoother"))
    assert [f.level for f in findings] == ["error"]
    assert "is GAMG but has no smoother" in findings[0].message


def test_div_check_warns_on_an_unbounded_scheme() -> None:
    pytest.importorskip("pybFoam")
    findings = check_div_scheme(_ctx(CASES / "unbounded_div"))
    assert [f.level for f in findings] == ["warning"]
    assert "unbounded" in findings[0].message


# -- required-files "present but does not load" -------------------------------


def test_required_files_errors_when_a_present_config_does_not_load() -> None:
    pytest.importorskip("pybFoam")
    from neofoam.framework.validation.checks import check_required_files
    from neofoam.mcp.registry import resolve_solver

    solver: Any = resolve_solver("incompressibleFluid")
    findings = check_required_files(
        CaseContext(case=CASES / "bad_controldict", solver=solver)
    )
    assert any(
        "controlDict" in f.file and "does not load" in f.message and f.level == "error"
        for f in findings
    )


# -- _text / _leaf_or_error seam ----------------------------------------------


def test_text_refuses_an_unreadable_leaf() -> None:
    with pytest.raises(TypeError):
        checks_mod._text(checks_mod.Unreadable(reason="x"))


def test_leaf_or_error_escalates_unreadable_and_passes_value() -> None:
    text, err = checks_mod._leaf_or_error(
        checks_mod.Unreadable(reason="bad"), "0/U", "type could not be read", fix="f"
    )
    assert text is None and err is not None and err.level == "error"
    text2, err2 = checks_mod._leaf_or_error(
        checks_mod.Value(text="symmetry"), "0/U", "m", fix="f"
    )
    assert text2 == "symmetry" and err2 is None
