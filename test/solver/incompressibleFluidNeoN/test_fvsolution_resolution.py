# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""How a field name is resolved against ``system/fvSolution`` entries.

Two lookups decide what the NeoN solver does with a field, and both accept an
OpenFOAM regex keyword:

* ``map_solver_settings`` — converts the ``solvers`` entry a field resolves to
  into its Ginkgo form, in place. ``create_fields`` calls it once per field of
  ``_MAPPED_SOLVER_DICTS``, so one regex entry is reached several times and must
  not be converted twice (the conversion consumes ``maxIter``).
* ``lookup_field_relaxation`` — the explicit pressure under-relaxation factor
  the PIMPLE/SIMPLE ``continuity`` operation blends with, from
  ``relaxationFactors.fields`` only.

Both are pure functions of a ``NeoN::Dictionary``: no mesh, no ``Foam::Time``,
no Kokkos initialization, so they run in-process here. The dictionaries are
built with the NeoN API because NeoFOAM exposes no Python-level OpenFOAM->NeoN
conversion; the entries mirror ``cases/regexSolverKeys/system/fvSolution``, and
the on-disk counterpart is covered by a real run in
``test_regex_fvsolution_keys``.
"""

from __future__ import annotations

from typing import Any

import neon._neon as nn  # NeoN Python bindings
import pytest

from neofoam import neofoam_bindings as nfb  # NeoFOAM Python bindings

# The committed case writes ``maxIter 1000`` for the regex entry, which is also
# the iteration count the conversion inserts by default — a re-converted entry
# would then be indistinguishable from a correctly converted one. 250 keeps the
# difference observable.
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
    """Each field the pattern names selects — and converts — that entry.

    ``reportName`` is the label the per-solve residual report prints, so it says
    *which* entry was converted, not merely that something was.
    """
    solvers = _solvers()

    nfb.map_solver_settings(solvers, field)

    assert solvers.subDict(_REGEX_SOLVER_KEY).get_string("reportName") == "Ilu+Bicgstab"


def test_literal_solver_entry_is_untouched_by_a_regex_field() -> None:
    """Converting the entry U resolves to leaves the literal p entry alone."""
    solvers = _solvers()

    nfb.map_solver_settings(solvers, "U")

    assert solvers.subDict("p").get_string("solver") == "PCG"
    assert not solvers.subDict("p").contains("reportName")


def test_one_entry_reached_by_several_fields_is_converted_once() -> None:
    """U, k and epsilon share an entry; the second and third calls must no-op.

    The conversion moves ``maxIter`` into ``criteria.iteration`` and removes it,
    so a second pass would silently reset the iteration limit to the default
    1000 — the failure the ``reportName`` stamp guards against.
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
    """On the final iteration ``pFinal`` wins, bare integer and all.

    OpenFOAM cases write ``pFinal 1;`` — an int-typed entry that a plain scalar
    read would reject; the lookup coerces it.
    """
    fv_solution = _fv_solution_with_relaxation()

    assert nfb.lookup_field_relaxation(fv_solution, "p", True) == 1.0


@pytest.mark.parametrize("final_iter", [False, True])
def test_field_relaxation_ignores_the_equations_entry(final_iter: bool) -> None:
    """U is relaxed as an *equation* (the momentum matrix), never as a field.

    Reading the ``equations`` factor here too would relax U twice; the two
    sub-dicts are independent, and OpenFOAM honours both.
    """
    fv_solution = _fv_solution_with_relaxation()

    assert nfb.lookup_field_relaxation(fv_solution, "U", final_iter) == 1.0


def test_field_relaxation_without_an_entry_is_a_no_op() -> None:
    """A field nothing covers relaxes by 1.0 — OpenFOAM's "no entry, no relax"."""
    fv_solution = _fv_solution_with_relaxation()

    assert nfb.lookup_field_relaxation(fv_solution, "k", False) == 1.0


def test_field_relaxation_with_a_malformed_relaxationfactors_is_a_no_op() -> None:
    """A ``relaxationFactors`` that is not a dictionary degrades, it does not throw.

    Without the isDict guard the lookup casts a string to a sub-dictionary and
    raises out of a solver step, so a typo in fvSolution would kill the run
    where OpenFOAM would only ignore it.
    """
    fv_solution = nn.Dictionary()
    fv_solution.insert_string("relaxationFactors", "0.3")

    assert nfb.lookup_field_relaxation(fv_solution, "p", False) == 1.0
