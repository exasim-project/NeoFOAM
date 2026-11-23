# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Spec for the typed ``fvSolution`` slices built on the linear-solver plugins."""

import shutil
from pathlib import Path
from typing import Any

import pybFoam as pyf
import pytest
from linear_solvers._helpers import (
    _CASE,
    _slice,
)

from neofoam.foam import (
    LinearSolverConversionWarning,
    neonFvSolution,
)
from neofoam.foam.fv_configs import SolverControls
from neofoam.framework.model import Model
from neofoam.io import write_configs
from neofoam.io.strategies.openfoam_strategy import _to_python
from neofoam.solver.incompressibleFluidNeoN.models.pressure_velocity.pimpleAlgorithm import (
    PimpleNeoNFvSolution,
)
from neofoam.solver.incompressibleFluidNeoN.models.pressure_velocity.simpleAlgorithm import (
    SimpleNeoNFvSolution,
)


def test_existing_case_is_converted_on_save(tmp_path: Path) -> None:
    case = tmp_path / "case"
    shutil.copytree(_CASE, case)
    with pytest.warns(LinearSolverConversionWarning):
        config = _slice("ConvertOnSave").load(case_dir=case)
    write_configs([config], case)
    written = _to_python(pyf.dictionary.read(str(case / "system/fvSolution")))["solvers"]
    assert written["p"] == {
        "solver": "Cg",
        "preconditioner": "Jacobi",
        "tolerance": "1e-06",
        "relTol": "0.05",
    }


def test_open_field_is_unchanged_on_save(tmp_path: Path) -> None:
    case = tmp_path / "case"
    shutil.copytree(_CASE, case)
    with pytest.warns(LinearSolverConversionWarning):
        config = _slice("OpenOnSave").load(case_dir=case)
    write_configs([config], case)
    written = _to_python(pyf.dictionary.read(str(case / "system/fvSolution")))["solvers"]
    assert written["k"] == {
        "solver": "smoothSolver",
        "smoother": "symGaussSeidel",
        "customKey": "kept",
    }


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        (
            "p",
            {
                "solver": "Cg",
                "preconditioner": {"type": "Jacobi"},
                "tolerance": 1e-06,
                "relTol": 0.05,
            },
        ),
        (
            "pFinal",
            {
                "solver": "Cg",
                "preconditioner": {"type": "Jacobi"},
                "tolerance": 1e-06,
                "relTol": 0.0,
            },
        ),
        (
            "U",
            {
                "solver": "Ir",
                "preconditioner": {"type": "Jacobi"},
                "tolerance": 1e-08,
                "relTol": 0.1,
            },
        ),
        (
            "UFinal",
            {
                "solver": "Ir",
                "preconditioner": {"type": "Jacobi"},
                "tolerance": 1e-08,
                "relTol": 0.0,
            },
        ),
    ],
)
def test_form_defaults_are_simple_blocks(field: str, expected: dict[str, Any]) -> None:
    assert _slice(f"Defaults{field}").form_defaults()["solvers"][field] == expected


def test_open_fields_stay_open_on_neon_slice() -> None:
    slice_cls = Model("OpenTurbulence").config(neonFvSolution)
    slice_cls.add("k")
    assert slice_cls._pending_sections["solvers"]["k"][0] is SolverControls


@pytest.mark.parametrize(
    ("spec_slice", "section", "expected"),
    [
        (
            PimpleNeoNFvSolution,
            "PIMPLE",
            {
                "nOuterCorrectors": 1,
                "nCorrectors": 2,
                "nNonOrthogonalCorrectors": 0,
                "pRefCell": 0,
                "pRefValue": 0,
            },
        ),
        (
            SimpleNeoNFvSolution,
            "SIMPLE",
            {"nNonOrthogonalCorrectors": 0, "pRefCell": 0, "pRefValue": 0},
        ),
    ],
)
def test_declared_controls_keep_their_starter_values(
    spec_slice: Any, section: str, expected: dict[str, int]
) -> None:
    assert spec_slice.form_defaults()[section] == expected
