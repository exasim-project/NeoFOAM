# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Spec for the pressure-velocity control factory (solver-side, pybFoam)."""

from pathlib import Path

import pytest

pytest.importorskip("pybFoam")

import pybFoam as pyf

from neofoam.solver.incompressibleFluid.models.pressure_velocity.control_factory import (
    read_residual_control,
)

_CASES = Path(__file__).parent / "cases"


def _algo_dict(case: str, algorithm: str):
    path = _CASES / case / "system" / "fvSolution"
    return pyf.dictionary.read(str(path)).subDict(algorithm)


def test_reads_simple_scalar_residual_control() -> None:
    rc = read_residual_control(_algo_dict("residualControl_simple", "SIMPLE"))
    assert rc["p"] == pytest.approx(1e-2)
    assert rc["U"] == pytest.approx(1e-3)
    assert rc["(k|epsilon)"] == pytest.approx(1e-3)


def test_reads_pimple_subdict_residual_control() -> None:
    rc = read_residual_control(_algo_dict("residualControl_pimple", "PIMPLE"))
    assert rc == {"(U|k|epsilon)": pytest.approx(1e-4)}


def test_absent_residual_control_is_empty() -> None:
    # The PIMPLE fixture's `solvers` subdict has no residualControl.
    assert read_residual_control(_algo_dict("residualControl_pimple", "solvers")) == {}


def test_read_residual_control_rejects_a_subdict_without_tolerance() -> None:
    d = pyf.dictionary.read(
        str(_CASES / "residualControl_malformed" / "system" / "fvSolution")
    )
    with pytest.raises(ValueError, match="tolerance"):
        read_residual_control(d.subDict("SIMPLE"))
