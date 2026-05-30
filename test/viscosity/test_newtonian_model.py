# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for the native ``Newtonian`` viscosity model — function + config (R5).

The model is built the way the solver builds it: selected from its case and fed
``nu`` read from the real ``constant/transportProperties`` (never a fabricated
constructor argument). Expected values come from the case's ``expected.yaml``.
"""

from pathlib import Path
from typing import Any

from neofoam.viscosity.base import ViscosityModel
from neofoam.viscosity.config import TransportPropertiesConfig

from viscosity.conftest import build_as_solver, case_for

#: Point the Newtonian model at the case the solver would feed it.
NEWTONIAN = case_for("Newtonian")


# --- function: runtime behaviour, built from the loaded dict ---
def test_nu_returns_value_from_dict() -> None:
    model = build_as_solver(NEWTONIAN)
    assert model.nu().value() == NEWTONIAN.model["nu"]


def test_correct_is_a_noop() -> None:
    assert build_as_solver(NEWTONIAN).correct() is None


def test_satisfies_viscosity_protocol() -> None:
    model: Any = build_as_solver(NEWTONIAN)
    assert isinstance(model, ViscosityModel)


# --- config: the model's own config loads, validates, and round-trips ---
def test_config_loads_and_validates() -> None:
    cfg = TransportPropertiesConfig.load(case_dir=NEWTONIAN.path)
    assert cfg.transportModel == NEWTONIAN.config["transportModel"]
    assert cfg.nu == NEWTONIAN.config["nu"]


def test_config_round_trips(tmp_path: Path) -> None:
    cfg = TransportPropertiesConfig.load(case_dir=NEWTONIAN.path)
    cfg.save(case_dir=tmp_path)
    reloaded = TransportPropertiesConfig.load(case_dir=tmp_path)
    assert reloaded.model_dump() == cfg.model_dump()
