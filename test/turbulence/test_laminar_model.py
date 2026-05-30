# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for the native ``laminar`` turbulence model — function + config (R5).

The model is built the way the solver builds it: selected from its case and
handed a viscosity model read from the same case's ``constant/transportProperties``
(never a fabricated ``nu``). Expected values come from the case's
``expected.yaml``. The viscous-stress *assembly* (the symbolic expression) is
covered by ``test_stress.py``; here we verify the model's own surface and that it
delegates to / reduces through its default stress model.
"""

from types import SimpleNamespace
from typing import Any

from neofoam.turbulence.base import TurbulenceModel
from neofoam.turbulence.config import TurbulencePropertiesConfig
from neofoam.turbulence.stress import LinearViscoStress

from turbulence.conftest import build_as_solver, case_for

#: Point the laminar model at the case the solver would feed it.
LAMINAR = case_for("laminar")


# --- function: runtime behaviour, built as the solver builds it ---
def test_nut_is_zero_for_laminar() -> None:
    assert build_as_solver(LAMINAR).nut() == LAMINAR.model["nut"]


def test_nu_delegates_to_viscosity_from_dict() -> None:
    assert build_as_solver(LAMINAR).nu().value() == LAMINAR.model["nu"]


def test_nuEff_reduces_to_molecular_viscosity() -> None:
    # nut == 0 ⇒ nuEff == nu; the default stress model is wired to the viscosity.
    assert build_as_solver(LAMINAR).nuEff().value() == LAMINAR.model["nu"]


def test_correct_is_a_noop() -> None:
    assert build_as_solver(LAMINAR).correct() is None


def test_default_stress_is_linear_visco_stress() -> None:
    assert isinstance(build_as_solver(LAMINAR).stress, LinearViscoStress)


def test_divDevReff_dispatches_to_injected_stress() -> None:
    """divDevReff is delegated to the stress model, not assembled in the model."""
    sentinel = object()
    stress = SimpleNamespace(divDevReff=lambda U: sentinel, nuEff=lambda: None)
    model = build_as_solver(LAMINAR)
    model.stress = stress
    assert model.divDevReff("U") is sentinel


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
