# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Runtime selection of the incompressibleVoFNeoN alpha-advection scheme (``advectionModel.py``).

The cases are ``incompressibleVoF``'s selection cases: a case must name the same scheme in both
solvers. MULES is the only member here, so an isoAdvector case must fail by name rather than run
MULES, and must select a scheme added through ``register``. The "unknown" and "explicit MULES"
variants are written into ``tmp_path`` through pybFoam's own dictionary reader/writer.
"""

import shutil
from pathlib import Path

import pytest
from pybFoam import dictionary

from neofoam.solver.incompressibleVoFNeoN.models.alpha_advection import (
    advectionModel as advection_module,
)
from neofoam.solver.incompressibleVoFNeoN.models.alpha_advection import (
    advectionModelNeoN,
    mulesNeoN,
)

_CASES = Path(__file__).parents[3] / "incompressibleVoF/models/alpha_advection/cases"


def _copy_with(tmp_path: Path, case_name: str, advection_scheme: str) -> Path:
    """``case_name`` in ``tmp_path`` with ``advectionScheme`` set."""
    shutil.copytree(_CASES / case_name, tmp_path, dirs_exist_ok=True)
    fv_solution_path = tmp_path / "system" / "fvSolution"
    fv_solution = dictionary.read(str(fv_solution_path))
    fv_solution.set("advectionScheme", advection_scheme)
    fv_solution.write(str(fv_solution_path))
    return tmp_path


def test_registered_names_start_with_mules() -> None:
    assert advectionModelNeoN.registered_names() == ["MULES"]


def test_detect_and_create_selects_mules_from_the_case(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(_CASES / "damBreak_mules")
    assert advectionModelNeoN.detect_and_create() is mulesNeoN


def test_detect_and_create_defaults_to_mules_without_fvSolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert advectionModelNeoN.detect_and_create() is mulesNeoN


def test_detect_and_create_explicit_mules_overrides_the_alpha_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(_copy_with(tmp_path, "damBreak_isoAdvector", "MULES"))
    assert advectionModelNeoN.detect_and_create() is mulesNeoN


def test_detect_and_create_raises_for_an_unregistered_isoadvector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(_CASES / "damBreak_isoAdvector")
    with pytest.raises(ValueError, match="Unknown advectionScheme 'isoAdvector'"):
        advectionModelNeoN.detect_and_create()


def test_detect_and_create_selects_a_registered_scheme(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(advection_module, "_SCHEMES", dict(advection_module._SCHEMES))
    added_scheme = object()
    advectionModelNeoN.register("isoAdvector", added_scheme)
    monkeypatch.chdir(_CASES / "damBreak_isoAdvector")
    assert advectionModelNeoN.detect_and_create() is added_scheme
