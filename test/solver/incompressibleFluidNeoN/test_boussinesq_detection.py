# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The NeoN Boussinesq model is detected from ``beta`` and ``TRef`` of the case it runs in."""

import shutil
from pathlib import Path
from typing import Any

import pytest

from neofoam.solver.incompressibleFluidNeoN.models import incompressibleFluidNeoNModel

CASES = Path(__file__).parent / "cases"


def _detected(case: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """The optional models detected in ``case``, run from its directory, by name."""
    shutil.copytree(CASES / case, tmp_path / case)
    monkeypatch.chdir(tmp_path / case)
    return {m.spec.name: m for m in incompressibleFluidNeoNModel.detect_models()}


def test_boussinesq_is_detected_from_beta_and_tref(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    boussinesq = _detected("hotRoomLaminar", tmp_path, monkeypatch)["boussinesq"]

    assert (boussinesq.config.beta, boussinesq.config.TRef) == (3e-3, 300.0)


def test_boussinesq_is_not_detected_without_beta_and_tref(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert "boussinesq" not in _detected("turbulentBox", tmp_path, monkeypatch)
