# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""How a simple ``fvSolution`` block becomes the dictionary NeoN runs.

``build_neon_solver_settings`` feeds the existing mapper, so each case asserts on the
mapped dictionary. Dictionaries are built with the NeoN API, as the OpenFOAM conversion
delivers them, and run in-process: this is a pure function of a ``NeoN::Dictionary``.
"""

from pathlib import Path
from typing import Any

import neon._neon as nn  # NeoN Python bindings
import pytest
from pydantic import ValidationError

from neofoam import neofoam_bindings as nfb  # NeoFOAM Python bindings
from neofoam.foam import LinearSolverConversionWarning
from neofoam.solver.incompressibleFluidNeoN.solver_settings import build_neon_solver_settings

_CASE_DIR = Path("/case")


def _entry(solver: str, preconditioner: str) -> Any:
    """A simple block with the stopping controls every case shares."""
    entry = nn.Dictionary()
    entry.insert_string("solver", solver)
    entry.insert_string("preconditioner", preconditioner)
    entry.insert_double("tolerance", 1e-06)
    entry.insert_double("relTol", 0.05)
    entry.insert_int("maxIter", 250)
    return entry


def _mapped(solvers: Any, field: str = "p") -> Any:
    """``field`` after the run-start build and the existing mapper."""
    build_neon_solver_settings(solvers, _CASE_DIR)
    nfb.map_solver_settings(solvers, field)
    return solvers.subDict(field)


def _solvers_with(field: str, entry: Any) -> Any:
    """A ``solvers`` dictionary holding one entry."""
    solvers = nn.Dictionary()
    solvers.insert_dict(field, entry)
    return solvers


@pytest.mark.parametrize(
    ("solver", "preconditioner", "solver_type", "preconditioner_type"),
    [
        ("Cg", "Ic", "solver::Cg", "preconditioner::Ic"),
        ("Cg", "Multigrid", "solver::Cg", "solver::Multigrid"),
        ("Bicg", "Ilu", "solver::Bicg", "preconditioner::Ilu"),
        ("Bicgstab", "Jacobi", "solver::Bicgstab", "preconditioner::Jacobi"),
        ("Gmres", "Ic", "solver::Gmres", "preconditioner::Ic"),
        ("Ir", "Jacobi", "solver::Ir", "preconditioner::Jacobi"),
    ],
)
def test_simple_block_maps_to_ginkgo_types(
    solver: str, preconditioner: str, solver_type: str, preconditioner_type: str
) -> None:
    mapped = _mapped(_solvers_with("p", _entry(solver, preconditioner)))
    selected = (mapped.get_string("type"), mapped.subDict("preconditioner").get_string("type"))
    assert selected == (solver_type, preconditioner_type)


def test_none_preconditioner_reaches_no_ginkgo_key() -> None:
    mapped = _mapped(_solvers_with("p", _entry("Cg", "none")))
    assert not mapped.contains("preconditioner")


@pytest.mark.parametrize(
    ("key", "expected"), [("absolute_residual_norm", 1e-06), ("initial_residual_norm", 0.05)]
)
def test_controls_become_ginkgo_norms(key: str, expected: float) -> None:
    mapped = _mapped(_solvers_with("p", _entry("Cg", "Ic")))
    assert mapped.subDict("criteria").get_double(key) == expected


def test_max_iter_becomes_the_iteration_cap() -> None:
    mapped = _mapped(_solvers_with("p", _entry("Cg", "Ic")))
    assert mapped.subDict("criteria").get_int("iteration") == 250


def test_l1_scaled_residual_is_on() -> None:
    mapped = _mapped(_solvers_with("p", _entry("Cg", "Ic")))
    assert mapped.get_bool("l1ScaledResidual") is True


def test_multigrid_tuning_reaches_the_mapper() -> None:
    entry = _entry("Cg", "Multigrid")
    tuning = nn.Dictionary()
    tuning.insert_string("type", "Multigrid")
    tuning.insert_int("nCellsInCoarsestLevel", 17)
    entry.remove("preconditioner")
    entry.insert_dict("preconditioner", tuning)
    mapped = _mapped(_solvers_with("p", entry))
    assert mapped.subDict("preconditioner").get_int("min_coarse_rows") == 17


def test_openfoam_block_is_left_to_the_mapper() -> None:
    entry = _entry("PCG", "DIC")
    mapped = _mapped(_solvers_with("U", entry), "U")
    assert mapped.get_string("reportName") == "Ic+Cg"


@pytest.mark.parametrize("field", ["p", "pFinal"])
def test_openfoam_dic_on_pressure_maps_to_jacobi(field: str) -> None:
    solvers = _solvers_with(field, _entry("PCG", "DIC"))
    with pytest.warns(LinearSolverConversionWarning, match="DIC on pressure"):
        mapped = _mapped(solvers, field)
    assert mapped.get_string("reportName") == "Jacobi+Cg"


def test_regex_key_resolving_to_pressure_maps_to_jacobi() -> None:
    solvers = _solvers_with('"p.*"', _entry("PCG", "DIC"))
    with pytest.warns(LinearSolverConversionWarning, match="DIC on pressure"):
        mapped = _mapped(solvers)
    assert mapped.get_string("reportName") == "Jacobi+Cg"


def test_regex_key_shared_with_velocity_keeps_its_mapping() -> None:
    solvers = _solvers_with('"(p|U)"', _entry("PCG", "DIC"))
    with pytest.warns(LinearSolverConversionWarning, match="DIC on pressure"):
        _mapped(solvers)
    nfb.map_solver_settings(solvers, "U")
    assert solvers.subDict('"(p|U)"').get_string("reportName") == "Ic+Cg"


def test_explicit_ic_on_pressure_stays_ic() -> None:
    mapped = _mapped(_solvers_with("p", _entry("Cg", "Ic")))
    assert mapped.subDict("preconditioner").get_string("type") == "preconditioner::Ic"


def test_pressure_block_without_preconditioner_runs_jacobi() -> None:
    entry = _entry("Bicgstab", "none")
    entry.remove("preconditioner")
    mapped = _mapped(_solvers_with("p", entry))
    assert mapped.subDict("preconditioner").get_string("type") == "preconditioner::Jacobi"


def test_regex_key_is_built_once() -> None:
    solvers = _solvers_with('"(U|UFinal)"', _entry("Bicgstab", "Ilu"))
    build_neon_solver_settings(solvers, _CASE_DIR)
    assert solvers.subDict('"(U|UFinal)"').get_string("type") == "solver::Bicgstab"


def test_relative_config_file_is_anchored_at_the_case() -> None:
    entry = nn.Dictionary()
    entry.insert_string("solver", "Ginkgo")
    entry.insert_string("configFile", "gko.json")
    solvers = _solvers_with("p", entry)
    build_neon_solver_settings(solvers, _CASE_DIR)
    assert solvers.subDict("p").get_string("configFile") == "/case/gko.json"


@pytest.mark.parametrize(("solver", "preconditioner"), [("Bicg", "Multigrid"), ("Ir", "none")])
def test_invalid_simple_block_fails_at_start(solver: str, preconditioner: str) -> None:
    solvers = _solvers_with("p", _entry(solver, preconditioner))
    with pytest.raises(ValidationError, match=preconditioner):
        build_neon_solver_settings(solvers, _CASE_DIR)


@pytest.mark.parametrize("field", ["U", "UFinal"])
def test_gmres_on_a_vector_field_fails_at_start(field: str) -> None:
    solvers = _solvers_with(field, _entry("Gmres", "Jacobi"))
    with pytest.raises(ValueError, match="Gmres cannot solve U"):
        build_neon_solver_settings(solvers, _CASE_DIR)


def _preconditioner(values: dict[str, Any]) -> Any:
    """A typed preconditioner sub-dictionary, as the file nests it."""
    node = nn.Dictionary()
    inserters = {str: node.insert_string, int: node.insert_int, float: node.insert_double}
    inserters[bool] = node.insert_bool
    for key, value in values.items():
        if isinstance(value, dict):
            node.insert_dict(key, _preconditioner(value))
        else:
            inserters[type(value)](key, value)
    return node


def _built_multigrid(values: dict[str, Any]) -> Any:
    """The ``p`` block after the run-start build, before the mapper."""
    entry = _entry("Cg", "Multigrid")
    entry.remove("preconditioner")
    entry.insert_dict("preconditioner", _preconditioner({"type": "Multigrid", **values}))
    solvers = _solvers_with("p", entry)
    build_neon_solver_settings(solvers, _CASE_DIR)
    return solvers.subDict("p")


def test_multigrid_options_become_flat_solver_keys() -> None:
    built = _built_multigrid(
        {"nCellsInCoarsestLevel": 50, "max_levels": 8, "cycle": "f", "nCoarsestSweeps": 12}
    )
    flat = (
        built.get_string("preconditioner"),
        built.get_int("nCellsInCoarsestLevel"),
        built.get_int("max_levels"),
        built.get_string("cycle"),
        built.get_int("nCoarsestSweeps"),
    )
    assert flat == ("GAMG", 50, 8, "f", 12)


def test_coarsening_stays_a_subdictionary() -> None:
    coarsening = {
        "max_iterations": 20,
        "max_unassigned_ratio": 0.1,
        "deterministic": True,
        "skip_sorting": False,
    }
    built = _built_multigrid({"coarsening": coarsening}).subDict("coarsening")
    read = (
        built.get_int("max_iterations"),
        built.get_double("max_unassigned_ratio"),
        built.get_bool("deterministic"),
        built.get_bool("skip_sorting"),
    )
    assert read == (20, 0.1, True, False)


@pytest.mark.parametrize(
    "key",
    ["max_levels", "cycle", "nPreSweeps", "nCoarsestSweeps", "coarsening"],
)
def test_unset_multigrid_option_is_absent(key: str) -> None:
    assert not _built_multigrid({}).contains(key)


@pytest.mark.parametrize(
    ("solver", "preconditioner", "key", "value"),
    [("Gmres", "Ic", "krylov_dim", 40), ("Ir", "Jacobi", "relaxation_factor", 0.8)],
)
def test_solver_option_is_a_flat_key(
    solver: str, preconditioner: str, key: str, value: Any
) -> None:
    entry = _entry(solver, preconditioner)
    (entry.insert_int if isinstance(value, int) else entry.insert_double)(key, value)
    solvers = _solvers_with("p", entry)
    build_neon_solver_settings(solvers, _CASE_DIR)
    getter = (
        solvers.subDict("p").get_int if isinstance(value, int) else solvers.subDict("p").get_double
    )
    assert getter(key) == value


def test_jacobi_block_size_is_a_flat_key() -> None:
    entry = _entry("Bicgstab", "Jacobi")
    entry.remove("preconditioner")
    entry.insert_dict("preconditioner", _preconditioner({"type": "Jacobi", "max_block_size": 4}))
    solvers = _solvers_with("p", entry)
    build_neon_solver_settings(solvers, _CASE_DIR)
    built = solvers.subDict("p")
    assert (built.get_string("preconditioner"), built.get_int("max_block_size")) == ("diagonal", 4)


@pytest.mark.parametrize(
    ("word", "expected"), [("yes", True), ("true", True), ("no", False), ("false", False)]
)
def test_switch_word_becomes_a_boolean(word: str, expected: bool) -> None:
    built = _built_multigrid({"coarsening": {"deterministic": word}})
    assert built.subDict("coarsening").get_bool("deterministic") is expected
