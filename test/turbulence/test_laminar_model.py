# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for the native ``laminar`` momentum-transport model.

Laminar flow has no eddy viscosity, so the model registers **no ``nut`` field**
and solves **no transport equation** — it contributes **no per-step operation**.
It owns the viscous stress: its ``@build`` registers the ``viscousStress`` object
the momentum equation resolves (and refreshes via ``update`` where it is consumed).
It is built the way the solver builds it (selected from its case, wrapped by
``SpecMomentumTransport``); expected values come from the case's ``expected.yaml``.
The stress assembly is exercised end-to-end by ``test_laminar_comparison``.
"""

from typing import Any

from neofoam.turbulence.config import TurbulencePropertiesConfig
from neofoam.turbulence.models.laminar import laminar

from turbulence.conftest import build_as_solver, case_for

#: Point the laminar model at the case the solver would feed it.
LAMINAR = case_for("laminar")


# --- laminar solves no transport equation, so it adds no per-step operation;
#     the momentum predictor refreshes the stress where it is consumed ---
def test_contributes_no_operation() -> None:
    assert list(build_as_solver(LAMINAR).operations) == []


# --- the model itself defines the (linear) stress it uses ---
def test_stress_kind_is_linear() -> None:
    assert build_as_solver(LAMINAR).stress_kind == "linear"


def test_model_registers_viscous_stress() -> None:
    # laminar's @build registers the viscousStress the momentum equation resolves.
    # That it is a LinearViscousStress and its assembly are covered end-to-end by
    # test_laminar_comparison.
    runtime = laminar.instantiate(LAMINAR.path)
    names = [s.name for s in runtime.run_build()]
    assert "models.viscousStress" in names


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
