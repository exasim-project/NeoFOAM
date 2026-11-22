# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""How a field name is resolved against ``system/fvSolution`` entries.

Both lookups accept an OpenFOAM regex keyword. ``map_solver_settings`` converts
the ``solvers`` entry a field resolves to into its Ginkgo form in place, and
``create_fields`` calls it once per mapped field — so one regex entry is reached
several times and must not be converted twice (the conversion consumes
``maxIter``). ``lookup_field_relaxation`` reads the pressure under-relaxation
factor from ``relaxationFactors.fields`` only.

Both are pure functions of a ``NeoN::Dictionary``, so they run in-process. The
dictionaries are built with the NeoN API — NeoFOAM exposes no Python-level
OpenFOAM->NeoN conversion — and mirror ``cases/regexSolverKeys``, whose on-disk
behaviour ``test_regex_fvsolution_keys`` covers with a real run.
"""

from __future__ import annotations

from typing import Any

import neon._neon as nn  # NeoN Python bindings
import pytest

from neofoam import neofoam_bindings as nfb  # NeoFOAM Python bindings

# The committed case's ``maxIter 1000`` is also the count the conversion inserts by
# default, so a re-converted entry would be indistinguishable; 250 keeps it visible.
_REGEX_MAX_ITER = 250

_REGEX_SOLVER_KEY = '"(U|k|epsilon)"'


def _solver_entry(solver: str, preconditioner: str, max_iter: int) -> Any:
    """One ``solvers`` sub-entry as the OpenFOAM->NeoN conversion delivers it."""
    entry = nn.Dictionary()
    entry.insert_string("solver", solver)
    entry.insert_string("preconditioner", preconditioner)
    entry.insert_double("tolerance", 1e-13)
    entry.insert_double("relTol", 1e-13)
    entry.insert_int("maxIter", max_iter)
    return entry


def _solvers() -> Any:
    """The ``solvers`` dict of ``cases/regexSolverKeys``: p literal, the rest regex."""
    solvers = nn.Dictionary()
    solvers.insert_dict("p", _solver_entry("PCG", "diagonal", 5000))
    solvers.insert_dict(_REGEX_SOLVER_KEY, _solver_entry("PBiCGStab", "DILU", _REGEX_MAX_ITER))
    return solvers


def _fv_solution_with_relaxation() -> Any:
    """``relaxationFactors`` of ``cases/regexSolverKeys``: equations and fields."""
    equations = nn.Dictionary()
    equations.insert_double(_REGEX_SOLVER_KEY, 0.7)
    equations.insert_int("UFinal", 1)

    fields = nn.Dictionary()
    fields.insert_double("p", 0.3)
    fields.insert_int("pFinal", 1)

    relaxation_factors = nn.Dictionary()
    relaxation_factors.insert_dict("equations", equations)
    relaxation_factors.insert_dict("fields", fields)

    fv_solution = nn.Dictionary()
    fv_solution.insert_dict("relaxationFactors", relaxation_factors)
    return fv_solution


@pytest.mark.parametrize("field", ["U", "k", "epsilon"])
def test_regex_solver_key_resolves_for_every_field_it_covers(field: str) -> None:
    """Each field the pattern names converts that entry; ``reportName`` says which."""
    solvers = _solvers()

    nfb.map_solver_settings(solvers, field)

    assert solvers.subDict(_REGEX_SOLVER_KEY).get_string("reportName") == "Ilu+Bicgstab"


def test_literal_solver_entry_is_untouched_by_a_regex_field() -> None:
    """Converting the entry U resolves to leaves the literal p entry alone."""
    solvers = _solvers()

    nfb.map_solver_settings(solvers, "U")

    assert solvers.subDict("p").get_string("solver") == "PCG"
    assert not solvers.subDict("p").contains("reportName")


def _smooth_solver_entry(**extra: object) -> Any:
    """A tutorial ``smoothSolver`` entry: a smoother, no explicit preconditioner."""
    entry = nn.Dictionary()
    entry.insert_string("solver", "smoothSolver")
    entry.insert_string("smoother", "GaussSeidel")
    entry.insert_double("tolerance", 1e-8)
    entry.insert_double("relTol", 0.1)
    for key, value in extra.items():
        entry.insert_int(key, int(value))  # type: ignore[call-overload]
    return entry


def test_smoothsolver_sweep_count_is_dropped_with_its_smoother() -> None:
    """``nSweeps`` rides along with ``smoother``; Ginkgo aborts on any key it does not know."""
    solvers = nn.Dictionary()
    solvers.insert_dict("U", _smooth_solver_entry(nSweeps=2))

    nfb.map_solver_settings(solvers, "U")

    assert not solvers.subDict("U").contains("nSweeps")
    assert not solvers.subDict("U").contains("smoother")


def _gamg_entry() -> Any:
    """A stock tutorial ``GAMG`` pressure entry, tuning keys and all."""
    entry = nn.Dictionary()
    entry.insert_string("solver", "GAMG")
    entry.insert_string("smoother", "GaussSeidel")
    entry.insert_double("tolerance", 1e-6)
    entry.insert_double("relTol", 0.1)
    return entry


def test_gamg_maps_to_cg_preconditioned_by_pgm_multigrid() -> None:
    """GAMG becomes Ginkgo algebraic multigrid, CG-accelerated.

    GAMG is picked for the SPD pressure Laplacian, so the multigrid is the *preconditioner*
    of an outer CG rather than the solver — a bare multigrid solve is far less robust.
    """
    solvers = nn.Dictionary()
    solvers.insert_dict("p", _gamg_entry())

    nfb.map_solver_settings(solvers, "p")

    entry = solvers.subDict("p")
    assert entry.get_string("solver") == "Ginkgo"
    assert entry.get_string("type") == "solver::Cg"
    multigrid = entry.subDict("preconditioner")
    assert multigrid.get_string("type") == "solver::Multigrid"
    assert multigrid.subDict("mg_level").get_string("type") == "multigrid::Pgm"
    assert entry.get_string("reportName") == "Multigrid+Cg"


def test_gamg_as_a_preconditioner_leaves_the_outer_solver_alone() -> None:
    """``preconditioner GAMG`` only replaces the preconditioner; PCG stays the solver."""
    solvers = nn.Dictionary()
    entry = nn.Dictionary()
    entry.insert_string("solver", "PCG")
    entry.insert_string("preconditioner", "GAMG")
    entry.insert_double("tolerance", 1e-6)
    entry.insert_double("relTol", 0.1)
    solvers.insert_dict("p", entry)

    nfb.map_solver_settings(solvers, "p")

    mapped = solvers.subDict("p")
    assert mapped.get_string("type") == "solver::Cg"
    assert mapped.subDict("preconditioner").get_string("type") == "solver::Multigrid"


def test_ncells_in_coarsest_level_becomes_min_coarse_rows() -> None:
    """OpenFOAM stops coarsening at nCellsInCoarsestLevel; Ginkgo at min_coarse_rows."""
    solvers = nn.Dictionary()
    entry = _gamg_entry()
    entry.insert_int("nCellsInCoarsestLevel", 500)
    solvers.insert_dict("p", entry)

    nfb.map_solver_settings(solvers, "p")

    mapped = solvers.subDict("p")
    assert mapped.subDict("preconditioner").get_int("min_coarse_rows") == 500
    assert not mapped.contains("nCellsInCoarsestLevel")


def test_gamg_only_tuning_keys_are_dropped_from_the_entry() -> None:
    """Ginkgo's config check aborts on any key it does not know — none may survive."""
    solvers = nn.Dictionary()
    entry = _gamg_entry()
    entry.insert_string("agglomerator", "faceAreaPair")
    entry.insert_int("mergeLevels", 1)
    entry.insert_bool("cacheAgglomeration", True)
    solvers.insert_dict("p", entry)

    nfb.map_solver_settings(solvers, "p")

    mapped = solvers.subDict("p")
    for key in ("agglomerator", "mergeLevels", "cacheAgglomeration"):
        assert not mapped.contains(key)


def test_one_entry_reached_by_several_fields_is_converted_once() -> None:
    """U, k and epsilon share an entry; the second and third calls must no-op.

    The conversion moves ``maxIter`` into ``criteria.iteration`` and removes it, so
    a second pass would silently reset the limit to the default 1000.
    """
    solvers = _solvers()

    for field in ("U", "k", "epsilon"):
        nfb.map_solver_settings(solvers, field)

    criteria = solvers.subDict(_REGEX_SOLVER_KEY).subDict("criteria")
    assert criteria.get_int("iteration") == _REGEX_MAX_ITER


def test_field_relaxation_reads_the_fields_entry() -> None:
    """Off the final iteration, p relaxes by its ``relaxationFactors.fields`` value."""
    fv_solution = _fv_solution_with_relaxation()

    assert nfb.lookup_field_relaxation(fv_solution, "p", False) == 0.3


def test_field_relaxation_prefers_the_int_valued_final_entry() -> None:
    """On the final iteration ``pFinal`` wins; cases write it as an int the lookup coerces."""
    fv_solution = _fv_solution_with_relaxation()

    assert nfb.lookup_field_relaxation(fv_solution, "p", True) == 1.0


@pytest.mark.parametrize("final_iter", [False, True])
def test_field_relaxation_ignores_the_equations_entry(final_iter: bool) -> None:
    """U is relaxed as an *equation*, never as a field; reading both would relax it twice."""
    fv_solution = _fv_solution_with_relaxation()

    assert nfb.lookup_field_relaxation(fv_solution, "U", final_iter) == 1.0


def test_field_relaxation_without_an_entry_is_a_no_op() -> None:
    """A field nothing covers relaxes by 1.0 — OpenFOAM's "no entry, no relax"."""
    fv_solution = _fv_solution_with_relaxation()

    assert nfb.lookup_field_relaxation(fv_solution, "k", False) == 1.0


def test_field_relaxation_with_a_malformed_relaxationfactors_is_a_no_op() -> None:
    """A ``relaxationFactors`` that is not a dictionary degrades, it does not throw.

    Without the isDict guard a typo in fvSolution kills the run mid-solve, where
    OpenFOAM would only ignore it.
    """
    fv_solution = nn.Dictionary()
    fv_solution.insert_string("relaxationFactors", "0.3")

    assert nfb.lookup_field_relaxation(fv_solution, "p", False) == 1.0
