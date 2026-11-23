# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Spec for the Ginkgo solvers, preconditioners and block reader."""

import warnings
from pathlib import Path
from typing import Any

import pybFoam as pyf
import pytest
from pydantic import ValidationError

from linear_solvers._helpers import (
    _CASE,
    _FILE,
    _LEGACY_BLOCKS,
    _REJECTED,
    _SILENT_LEGACY,
    _SIMPLE_BLOCKS,
    _WARNED_KEYS,
    _blocks,
    _case_with_p,
    _entry,
    _slice,
    _validated,
)
from neofoam.foam import (
    GinkgoConfigFile,
    LinearSolverConversionWarning,
    linear_solver_entry,
)
from neofoam.io import write_configs
from neofoam.io.strategies.openfoam_strategy import _to_python


@pytest.mark.parametrize(("name", "plugin"), _SIMPLE_BLOCKS)
def test_simple_block_selects_its_plugin(name: str, plugin: str) -> None:
    entry = _entry().model_validate(_blocks("simpleBlocks")[name])
    assert (type(entry.root).__name__, entry.root.backend) == (plugin, "ginkgo")


@pytest.mark.parametrize(("name", "_plugin"), _SIMPLE_BLOCKS)
def test_simple_block_file_round_trip_keeps_keys_and_order(
    name: str, _plugin: str, tmp_path: Path
) -> None:
    block = _blocks("simpleBlocks")[name]
    slice_cls = _slice(f"RoundTrip{name}")
    case = _case_with_p(tmp_path, block)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LinearSolverConversionWarning)
        config = slice_cls.load(case_dir=case)
    write_configs([config], case)
    written = _to_python(pyf.dictionary.read(str(case / "system/fvSolution")))["solvers"]["p"]
    assert list(written.items()) == list(block.items())


@pytest.mark.parametrize(("name", "_plugin"), _SIMPLE_BLOCKS)
def test_backend_is_absent_from_dumps(name: str, _plugin: str) -> None:
    entry = _entry().model_validate(_blocks("simpleBlocks")[name])
    assert "backend" not in entry.model_dump(context=_FILE)


@pytest.mark.parametrize(("name", "expected"), _LEGACY_BLOCKS)
def test_existing_block_loads_as_ginkgo_equivalent(name: str, expected: dict[str, Any]) -> None:
    entry = _validated(_blocks("legacyBlocks")[name])
    assert entry.model_dump() == expected


@pytest.mark.parametrize(("name", "dropped"), _WARNED_KEYS)
def test_conversion_warns_naming_dropped_keys(name: str, dropped: list[str]) -> None:
    with pytest.warns(LinearSolverConversionWarning, match=repr(dropped).replace("[", r"\[")):
        _entry().model_validate(_blocks("legacyBlocks")[name])


@pytest.mark.parametrize("name", _SILENT_LEGACY)
def test_lossless_conversion_does_not_warn(name: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", LinearSolverConversionWarning)
        _entry().model_validate(_blocks("legacyBlocks")[name])


@pytest.mark.parametrize("name", _REJECTED)
def test_unsupported_block_is_rejected_naming_field(name: str) -> None:
    block = _blocks("rejectedBlocks")[name]
    slice_cls = _slice(f"Rejected{name}")
    data = _to_python(pyf.dictionary.read(str(_CASE / "system/fvSolution")))
    data["solvers"]["p"] = block
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LinearSolverConversionWarning)
        with pytest.raises(ValidationError) as caught:
            slice_cls.model_validate(data)
    assert caught.value.errors()[0]["loc"][:2] == ("solvers", "p")
    assert all(error["type"] != "missing" for error in caught.value.errors())


@pytest.mark.parametrize("field", ["U", "UFinal"])
def test_gmres_is_rejected_on_a_vector_field(field: str) -> None:
    slice_cls = _slice(f"VectorGmres{field}")
    data = _to_python(pyf.dictionary.read(str(_CASE / "system/fvSolution")))
    data["solvers"][field] = {"solver": "Gmres", "preconditioner": "Jacobi"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LinearSolverConversionWarning)
        with pytest.raises(ValidationError, match="Gmres cannot solve U") as caught:
            slice_cls.model_validate(data)
    assert caught.value.errors()[0]["loc"][:2] == ("solvers", field)


def test_gmres_is_accepted_on_pressure() -> None:
    slice_cls = _slice("ScalarGmres")
    data = _to_python(pyf.dictionary.read(str(_CASE / "system/fvSolution")))
    data["solvers"]["p"] = {"solver": "Gmres", "preconditioner": {"type": "Ic"}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LinearSolverConversionWarning)
        config = slice_cls.model_validate(data)
    assert config.solvers.p.root.solver == "Gmres"


@pytest.mark.parametrize(
    ("plugin", "default"), [("Cg", "none"), ("Bicg", "none"), ("Gmres", "none"), ("Ir", "Jacobi")]
)
def test_missing_preconditioner_is_written_explicitly(plugin: str, default: str) -> None:
    entry = _entry().model_validate({"solver": plugin})
    assert entry.model_dump(context=_FILE) == {"solver": plugin, "preconditioner": default}


@pytest.mark.parametrize("name", ["none", "Jacobi", "Ic", "Ilu", "Multigrid"])
def test_preconditioner_without_options_dumps_object_for_the_form(name: str) -> None:
    entry = _entry().model_validate({"solver": "Cg", "preconditioner": name})
    assert entry.model_dump()["preconditioner"] == {"type": name}


@pytest.mark.parametrize("name", ["none", "Jacobi", "Ic", "Ilu", "Multigrid"])
def test_preconditioner_without_options_is_a_word_in_the_file(name: str) -> None:
    entry = _entry().model_validate({"solver": "Cg", "preconditioner": {"type": name}})
    assert entry.model_dump(context=_FILE)["preconditioner"] == name


def test_preconditioner_with_options_is_a_dictionary_in_the_file() -> None:
    block = {"type": "Multigrid", "nCellsInCoarsestLevel": 50}
    entry = _entry().model_validate({"solver": "Cg", "preconditioner": block})
    assert entry.model_dump(context=_FILE)["preconditioner"] == block


@pytest.mark.parametrize(
    ("solver", "arms"),
    [
        ("Cg", {"none", "Jacobi", "Ic", "Ilu", "Multigrid"}),
        ("Bicg", {"none", "Jacobi", "Ic", "Ilu"}),
        ("Bicgstab", {"none", "Jacobi", "Ic", "Ilu"}),
        ("Gmres", {"none", "Jacobi", "Ic", "Ilu"}),
        ("Ir", {"Jacobi", "Ic", "Ilu"}),
    ],
)
def test_solver_schema_lists_only_compatible_preconditioners(solver: str, arms: set[str]) -> None:
    schema = _entry().model_json_schema()
    field = schema["$defs"][solver]["properties"]["preconditioner"]
    listed = {
        schema["$defs"][arm["$ref"].rsplit("/", 1)[1]]["properties"]["type"]["const"]
        for arm in field["oneOf"]
    }
    assert (listed, set(field["discriminator"]["mapping"])) == (arms, arms)


@pytest.mark.parametrize(
    ("solver", "preconditioner"), [("Bicg", "Multigrid"), ("Gmres", "Multigrid"), ("Ir", "none")]
)
def test_python_validation_still_rejects_what_the_schema_hides(
    solver: str, preconditioner: str
) -> None:
    with pytest.raises(ValidationError, match=f"{solver} with {preconditioner}"):
        _entry().model_validate({"solver": solver, "preconditioner": preconditioner})


def test_multigrid_arm_owns_the_tuning_options() -> None:
    arm = _entry().model_json_schema()["$defs"]["Multigrid"]["properties"]
    assert set(arm) == {
        "type",
        "nCellsInCoarsestLevel",
        "max_levels",
        "cycle",
        "nPreSweeps",
        "nCoarsestSweeps",
        "coarsening",
    }


def test_config_file_is_described_as_fallback() -> None:
    arm = _entry().model_json_schema()["$defs"]["GinkgoConfigFile"]
    assert "fallback" in arm["title"].lower()
    assert arm["description"].startswith("Fallback for Ginkgo settings")


def test_config_file_arm_has_no_default_path() -> None:
    assert "default" not in GinkgoConfigFile.model_json_schema()["properties"]["configFile"]


@pytest.mark.parametrize(
    ("definition", "option", "expected"),
    [
        ("Cg", "minIter", {"nfLevel": "advanced"}),
        ("Gmres", "krylov_dim", {"nfLevel": "advanced", "nfDefaultHint": 100}),
        ("Ir", "relaxation_factor", {"nfLevel": "advanced", "nfDefaultHint": 1}),
        ("Jacobi", "max_block_size", {"nfLevel": "advanced", "nfDefaultHint": 1}),
        (
            "Multigrid",
            "nCellsInCoarsestLevel",
            {"nfLevel": "advanced", "nfGroup": "Hierarchy", "nfDefaultHint": 64},
        ),
        (
            "Multigrid",
            "max_levels",
            {"nfLevel": "advanced", "nfGroup": "Hierarchy", "nfDefaultHint": 10},
        ),
        (
            "Multigrid",
            "cycle",
            {"nfLevel": "advanced", "nfGroup": "Hierarchy", "nfDefaultHint": "v"},
        ),
        (
            "Multigrid",
            "nPreSweeps",
            {"nfLevel": "advanced", "nfGroup": "Smoothing", "nfDefaultHint": 1},
        ),
        (
            "Multigrid",
            "nCoarsestSweeps",
            {"nfLevel": "advanced", "nfGroup": "Smoothing", "nfDefaultHint": 8},
        ),
        ("Multigrid", "coarsening", {"nfLevel": "advanced", "nfGroup": "Pgm coarsening"}),
        ("PgmCoarsening", "max_iterations", {"nfLevel": "advanced", "nfDefaultHint": 15}),
        ("PgmCoarsening", "max_unassigned_ratio", {"nfLevel": "advanced", "nfDefaultHint": 0.05}),
        ("PgmCoarsening", "deterministic", {"nfLevel": "advanced", "nfDefaultHint": False}),
        ("PgmCoarsening", "skip_sorting", {"nfLevel": "advanced", "nfDefaultHint": False}),
    ],
)
def test_option_schema_carries_its_metadata(
    definition: str, option: str, expected: dict[str, Any]
) -> None:
    prop = _entry().model_json_schema()["$defs"][definition]["properties"][option]
    assert {key: value for key, value in prop.items() if key.startswith("nf")} == expected


@pytest.mark.parametrize("definition", ["Gmres", "Ir", "Jacobi", "Multigrid", "PgmCoarsening"])
def test_option_schema_has_no_null_default(definition: str) -> None:
    props = _entry().model_json_schema()["$defs"][definition]["properties"]
    assert [key for key, prop in props.items() if prop.get("default", 0) is None] == []


@pytest.mark.parametrize(
    "block",
    [
        {"solver": "Gmres", "preconditioner": "Ic"},
        {"solver": "Ir", "preconditioner": "Jacobi"},
        {"solver": "Cg", "preconditioner": {"type": "Jacobi"}},
        {"solver": "Cg", "preconditioner": {"type": "Multigrid"}},
        {"solver": "Cg", "preconditioner": {"type": "Multigrid", "coarsening": {}}},
    ],
)
def test_unset_options_are_not_written(block: dict[str, Any]) -> None:
    dumped = _entry().model_validate(block).model_dump(context=_FILE)
    assert set(dumped) == {"solver", "preconditioner"} and isinstance(dumped["preconditioner"], str)


@pytest.mark.parametrize(
    ("preconditioner", "expected"),
    [
        ({"type": "Ic"}, {}),
        ({"type": "Multigrid"}, {}),
        ({"type": "Multigrid", "coarsening": {}}, {}),
        ({"type": "Jacobi", "max_block_size": 4}, {"max_block_size": 4}),
        ({"type": "Multigrid", "nCellsInCoarsestLevel": 50}, {"nCellsInCoarsestLevel": 50}),
        ({"type": "Multigrid", "max_levels": 8}, {"max_levels": 8}),
        ({"type": "Multigrid", "cycle": "f"}, {"cycle": "f"}),
        ({"type": "Multigrid", "nPreSweeps": 2}, {"nPreSweeps": 2}),
        ({"type": "Multigrid", "nCoarsestSweeps": 12}, {"nCoarsestSweeps": 12}),
        (
            {"type": "Multigrid", "coarsening": {"max_iterations": 20}},
            {"coarsening": {"max_iterations": 20}},
        ),
        (
            {"type": "Multigrid", "coarsening": {"max_unassigned_ratio": 0.1}},
            {"coarsening": {"max_unassigned_ratio": 0.1}},
        ),
        (
            {"type": "Multigrid", "coarsening": {"deterministic": False}},
            {"coarsening": {"deterministic": False}},
        ),
        (
            {"type": "Multigrid", "coarsening": {"skip_sorting": True}},
            {"coarsening": {"skip_sorting": True}},
        ),
    ],
)
def test_mapper_entries_hold_the_set_options(
    preconditioner: dict[str, Any], expected: dict[str, Any]
) -> None:
    entry = _entry().model_validate({"solver": "Cg", "preconditioner": preconditioner})
    assert entry.root.preconditioner.mapper_entries() == expected


@pytest.mark.parametrize("field", ["p", "pFinal"])
def test_openfoam_dic_on_pressure_loads_as_jacobi_with_a_warning(field: str) -> None:
    data = _to_python(pyf.dictionary.read(str(_CASE / "system/fvSolution")))
    data["solvers"][field] = _blocks("openfoamBlocks")["PcgDic"]
    with pytest.warns(LinearSolverConversionWarning, match="DIC on pressure converted to Jacobi"):
        config = _slice(f"PressureDic{field}").model_validate(data)
    assert getattr(config.solvers, field).root.preconditioner.type == "Jacobi"


def test_explicit_ic_on_pressure_stays_ic() -> None:
    entry = linear_solver_entry("ginkgo", "pressure")
    with warnings.catch_warnings():
        warnings.simplefilter("error", LinearSolverConversionWarning)
        block = entry.model_validate({"solver": "Cg", "preconditioner": "Ic"})
    assert block.root.preconditioner.type == "Ic"


def test_pressure_block_without_preconditioner_defaults_to_jacobi() -> None:
    block = linear_solver_entry("ginkgo", "pressure").model_validate({"solver": "Bicgstab"})
    assert block.root.preconditioner.type == "Jacobi"


def test_vector_block_without_preconditioner_keeps_its_default() -> None:
    block = linear_solver_entry("ginkgo", "momentum").model_validate({"solver": "Bicgstab"})
    assert block.root.preconditioner.type == "none"
