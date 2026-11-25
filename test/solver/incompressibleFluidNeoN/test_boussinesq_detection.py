# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The NeoN Boussinesq model is detected and loaded from the case directory it is given.

``beta``/``TRef`` in ``constant/transportProperties`` switch it on; ``cases/hotRoomLaminar``
has them, ``cases/turbulentBox`` does not. The working directory is elsewhere, as for a
region of a multiRun case (case directory ``<region>/``).
"""

import shutil
from pathlib import Path
from typing import Any

import pytest

from neofoam.solver.incompressibleFluidNeoN.models import incompressibleFluidNeoNModel

CASES = Path(__file__).parent / "cases"


def _detected(case: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    shutil.copytree(CASES / case, tmp_path / "fluid")
    monkeypatch.chdir(tmp_path)
    return {m.spec.name: m for m in incompressibleFluidNeoNModel.detect_models(Path("fluid"))}


def test_boussinesq_is_detected_and_loaded_from_the_case_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    boussinesq = _detected("hotRoomLaminar", tmp_path, monkeypatch)["boussinesq"]

    assert (boussinesq.config.beta, boussinesq.config.TRef) == (3e-3, 300.0)


def test_boussinesq_is_not_detected_without_beta_and_tref(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert "boussinesq" not in _detected("turbulentBox", tmp_path, monkeypatch)
