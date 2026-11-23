# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Spec for the OpenFOAM solver and preconditioner plugins."""

from pathlib import Path

import pybFoam as pyf
import pytest

from linear_solvers._helpers import (
    _blocks,
    _case_with_p,
    _slice,
)
from neofoam.foam import (
    linear_solver_entry,
    openfoamFvSolution,
)
from neofoam.io import write_configs
from neofoam.io.strategies.openfoam_strategy import _to_python


@pytest.mark.parametrize(
    "name",
    [
        "PcgDic",
        "PcgNestedGamg",
        "GamgUnmodelledKey",
        "SmoothSolver",
        "LibrarySolver",
        "UnknownPreconditioner",
        "SmootherDictionary",
        "NoPreconditioner",
        "CoupledVector",
    ],
)
def test_openfoam_block_file_round_trip_is_lossless(name: str, tmp_path: Path) -> None:
    block = _blocks("openfoamBlocks")[name]
    slice_cls = _slice(f"FoamRoundTrip{name}", openfoamFvSolution)
    case = _case_with_p(tmp_path, block)
    write_configs([slice_cls.load(case_dir=case)], case)
    written = _to_python(pyf.dictionary.read(str(case / "system/fvSolution")))["solvers"]["p"]
    assert list(written.items()) == list(block.items())


@pytest.mark.parametrize(
    ("role", "name", "plugin"),
    [
        ("pressure", "PcgDic", "FoamPCG"),
        ("momentum", "PcgDic", "FoamOtherBlock"),
        ("pressure", "PcgNestedGamg", "FoamPCG"),
        ("pressure", "LibrarySolver", "FoamOtherBlock"),
        ("pressure", "UnknownPreconditioner", "FoamOtherBlock"),
        ("pressure", "SmootherDictionary", "FoamOtherBlock"),
        ("momentum", "CoupledVector", "FoamOtherBlock"),
    ],
)
def test_openfoam_block_selects_typed_or_kept_arm(role: str, name: str, plugin: str) -> None:
    entry = linear_solver_entry("openfoam", role).model_validate(_blocks("openfoamBlocks")[name])
    assert type(entry.root).__name__ == plugin


@pytest.mark.parametrize(
    ("role", "expected"),
    [
        (
            "pressure",
            {
                "solver": ["PBiCG"],
                "preconditioner": ["DILU", "distributedDILU"],
                "smoother": ["DILU", "DILUGaussSeidel"],
            },
        ),
        (
            "momentum",
            {
                "solver": ["FPCG", "PCG", "PPCG", "PPCR"],
                "preconditioner": ["DIC", "FDIC", "distributedDIC"],
                "smoother": ["DIC", "DICGaussSeidel", "FDIC"],
            },
        ),
    ],
)
def test_openfoam_role_schema_lists_refused_names(
    role: str, expected: dict[str, list[str]]
) -> None:
    schema = linear_solver_entry("openfoam", role).model_json_schema()
    assert schema["nfRefusedChoices"] == expected


def test_nested_gamg_options_are_advanced() -> None:
    schema = linear_solver_entry("openfoam", "pressure").model_json_schema()
    props = schema["$defs"]["FoamGAMGPreconditioner"]["properties"]
    assert (props["smoother"]["nfLevel"], props["nCellsInCoarsestLevel"]["nfDefaultHint"]) == (
        "advanced",
        10,
    )
