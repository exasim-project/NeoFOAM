# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for the native ``Newtonian`` viscosity model — operation + config.

The model owns the molecular viscosity field ``nu``: it is built the way the
solver builds it (selected from its case) and its operation is run to publish
``fields.nu``, exactly as the solver merges the model's operations into the DAG.
Expected values come from the case's ``expected.yaml`` — never a fabricated
constructor argument.
"""

from pathlib import Path

from neofoam.framework.model import ModelRuntime
from neofoam.viscosity.config import TransportPropertiesConfig

from viscosity.conftest import build_as_solver, case_for, run_field_ops

#: Point the Newtonian model at the case the solver would feed it.
NEWTONIAN = case_for("Newtonian")


# --- model owns + updates its nu field, run as the solver runs it ---
def test_model_is_a_runtime() -> None:
    assert isinstance(build_as_solver(NEWTONIAN), ModelRuntime)


def test_update_nu_publishes_field_from_dict() -> None:
    fields = run_field_ops(build_as_solver(NEWTONIAN))
    assert fields["nu"].value() == NEWTONIAN.model["nu"]


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
