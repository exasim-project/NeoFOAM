# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for the native ``laminar`` momentum-transport model.

The model owns the eddy-viscosity field ``nut`` (registered/updated by its
operation) and registers the stress computer the small read interface dispatches
to. It is built the way the solver builds it (selected from its case, wrapped by
``SpecMomentumTransport``); expected values come from the case's ``expected.yaml``.
The viscous-stress assembly itself is exercised end-to-end by
``test_laminar_comparison``.
"""

from typing import Any

import pytest

from neofoam.turbulence.base import TurbulenceModel
from neofoam.turbulence.config import TurbulencePropertiesConfig
from neofoam.turbulence.momentumTransport import _STRESS
from neofoam.turbulence.stress import linear_viscous_stress

from turbulence.conftest import build_as_solver, case_for, run_field_ops

#: Point the laminar model at the case the solver would feed it.
LAMINAR = case_for("laminar")


# --- model owns + updates its nut field, run as the solver runs it ---
def test_update_nut_publishes_zero_field() -> None:
    fields = run_field_ops(build_as_solver(LAMINAR))
    assert fields["nut"].value() == LAMINAR.model["nut"]


# --- the model registers / dispatches its own stress computer ---
def test_registers_linear_viscous_stress() -> None:
    assert _STRESS["laminar"] is linear_viscous_stress


def test_divDevReff_dispatches_to_registered_stress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """divDevReff routes to the model's registered stress computer."""
    sentinel = object()
    monkeypatch.setitem(_STRESS, "laminar", lambda U, nu, nut: sentinel)
    model = build_as_solver(LAMINAR)
    assert model.divDevReff("U", "nu", "nut") is sentinel


def test_satisfies_turbulence_protocol() -> None:
    model: Any = build_as_solver(LAMINAR)
    assert isinstance(model, TurbulenceModel)


# --- config: the model's own config loads, validates, and round-trips ---
def test_config_loads_and_validates() -> None:
    cfg = TurbulencePropertiesConfig.load(case_dir=LAMINAR.path)
    assert cfg.simulationType == LAMINAR.config["simulationType"]
    assert cfg.RAS is None
    assert cfg.LES is None


def test_config_round_trips(tmp_path: Any) -> None:
    cfg = TurbulencePropertiesConfig.load(case_dir=LAMINAR.path)
    cfg.save(case_dir=tmp_path)
    reloaded = TurbulencePropertiesConfig.load(case_dir=tmp_path)
    assert reloaded.model_dump() == cfg.model_dump()
