# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Runtime selection of the incompressibleVoFNeoN interface model (``interfaceModel.py``).

The case is the solver suite's ``stationaryDroplet`` (its transportProperties carries no
``interfaceModel``); the keyed variants are written from it into ``tmp_path`` through
pybFoam's own dictionary reader/writer. Building a member needs the NeoN runtime, which the
solver comparisons exercise; these tests stop at the name and the class.
"""

import shutil
from pathlib import Path

import pytest
from pybFoam import dictionary

from neofoam.solver.incompressibleVoFNeoN.models.interface import (
    gradAlphaInterface,
    interfaceModelNeoN,
)

_CASE = Path(__file__).parents[2] / "cases" / "stationaryDroplet"


def _copy_with(tmp_path: Path, interface_model: str) -> Path:
    shutil.copytree(_CASE, tmp_path, dirs_exist_ok=True)
    path = tmp_path / "constant" / "transportProperties"
    transport = dictionary.read(str(path))
    transport.set("interfaceModel", interface_model)
    transport.write(str(path))
    return tmp_path


def test_registered_names_are_gradAlpha_only() -> None:
    assert interfaceModelNeoN.registered_names() == ["gradAlpha"]


def test_name_from_case_defaults_to_gradAlpha(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(_CASE)
    assert interfaceModelNeoN.name_from_case() == "gradAlpha"


def test_select_returns_the_member_the_case_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(_copy_with(tmp_path, "gradAlpha"))
    assert interfaceModelNeoN.select(interfaceModelNeoN.name_from_case()) is gradAlphaInterface


def test_select_raises_for_an_unknown_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(_copy_with(tmp_path, "heightFunction"))
    with pytest.raises(ValueError, match="Unknown interfaceModel 'heightFunction'"):
        interfaceModelNeoN.select(interfaceModelNeoN.name_from_case())
