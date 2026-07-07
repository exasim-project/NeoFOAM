# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for the native ``Newtonian`` viscosity model — build + config.

The model owns the molecular viscosity field ``nu``. For a Newtonian fluid ``nu``
is a constant, registered once at BUILD by the solver's ``create_fields`` from
``transportProperties``, so the model contributes **no per-step operation**. It is
built the way the solver builds it (selected from its case); the constant-``nu``
value is exercised end-to-end by ``test_laminar_comparison``. Config expectations
come from the case's ``expected.yaml`` — never a fabricated constructor argument.
"""

from pathlib import Path

from neofoam.framework.model import ModelRuntime
from neofoam.viscosity.config import TransportPropertiesConfig

from viscosity.conftest import build_as_solver, case_for

#: Point the Newtonian model at the case the solver would feed it.
NEWTONIAN = case_for("Newtonian")


# --- model is built as a runtime; nu is a build-time constant (no step op) ---
def test_model_is_a_runtime() -> None:
    assert isinstance(build_as_solver(NEWTONIAN), ModelRuntime)


def test_contributes_no_step_operation() -> None:
    assert list(build_as_solver(NEWTONIAN).operations) == []


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
