# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Runtime selection of the incompressibleVoFNeoN surface tension force (``forces.py``).

The case is the solver suite's ``stationaryDroplet`` (no ``surfaceTensionModel`` key); the
keyed variant is written from it into ``tmp_path`` through pybFoam's own dictionary
reader/writer. That ``sigma 0`` leaves the face force gravity alone, bit for bit, is the
``sigma 0`` damBreak comparisons' job: building the forces needs the NeoN runtime.
"""

import shutil
from pathlib import Path

import pytest
from pybFoam import dictionary

from neofoam.solver.incompressibleVoFNeoN.models.surface_forces import (
    brackbillForce,
    surface_tension_model,
    surface_tension_model_name,
)

_CASE = Path(__file__).parents[2] / "cases" / "stationaryDroplet"


def test_surface_tension_model_defaults_to_brackbill(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(_CASE)
    assert surface_tension_model(surface_tension_model_name()) is brackbillForce


def test_surface_tension_model_raises_for_an_unknown_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shutil.copytree(_CASE, tmp_path, dirs_exist_ok=True)
    path = tmp_path / "constant" / "transportProperties"
    transport = dictionary.read(str(path))
    transport.set("surfaceTensionModel", "sharpSurfaceForce")
    transport.write(str(path))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="Unknown surfaceTensionModel 'sharpSurfaceForce'"):
        surface_tension_model(surface_tension_model_name())
