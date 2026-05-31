# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for the native ``laminar`` momentum-transport model.

The model owns the eddy-viscosity field ``nut`` — but for laminar flow ``nut`` is
a constant zero, registered once at BUILD by the solver's ``create_fields``, so
the model contributes **no per-step operation**. It declares the ``stress_kind``
the ``viscousStress`` dispatch keys off. It is built the way the solver builds it
(selected from its case, wrapped by ``SpecMomentumTransport``); expected values
come from the case's ``expected.yaml``. The constant-``nut`` value and the stress
assembly are exercised end-to-end by ``test_laminar_comparison``.
"""

from typing import Any

from neofoam.turbulence.config import TurbulencePropertiesConfig
from neofoam.turbulence.stress import LinearViscousStress

from turbulence.conftest import build_as_solver, case_for

#: Point the laminar model at the case the solver would feed it.
LAMINAR = case_for("laminar")


# --- nut is a build-time constant, so the model adds no per-step operation ---
def test_contributes_no_step_operation() -> None:
    assert list(build_as_solver(LAMINAR).operations) == []


# --- the model itself defines the (linear) stress it uses ---
def test_stress_kind_is_linear() -> None:
    assert build_as_solver(LAMINAR).stress_kind == "linear"


def test_model_defines_linear_viscous_stress() -> None:
    stress = build_as_solver(LAMINAR).viscous_stress()
    assert isinstance(stress, LinearViscousStress)


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
