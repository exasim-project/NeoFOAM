# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for the ``laminar`` momentum-transport model (native NeoN + fallback).

After the family merge ``laminar`` is a dual-shape model in the single
``momentumTransportModel`` family:

* native NeoN (``fallback=False``): its ``@build`` emits ``nut = 0`` and the
  surface ``nuEff``; it solves no transport equation, so it declares **no native
  ``@operation``**.
* pybFoam fallback (``fallback=True``): one co-located ``fallback=True``
  ``laminarCorrect`` op advances the wrapped pybFoam model.

Structure is checked without a live NeoN runtime; the end-to-end NeoN ``nut``/
``nuEff`` values are covered bit-for-bit by ``test_neon_turbulence_parity``.
Expected config values come from the case's ``expected.yaml``.
"""

from typing import Any

from neofoam.turbulence.config import TurbulencePropertiesConfig
from neofoam.turbulence.models.laminar import laminar

from turbulence.conftest import case_for

#: Point the laminar model at the case the solver would feed it.
LAMINAR = case_for("laminar")


def _runtime() -> Any:
    return laminar.instantiate(LAMINAR.path)


# --- laminar solves no transport equation, so it declares no NATIVE operation ---
def test_declares_no_native_operation() -> None:
    assert _runtime().native_operations() == []


# --- but it declares one fallback correct op for the pybFoam path ---
def test_declares_one_fallback_correct_op() -> None:
    ops = _runtime().fallback_operations()
    assert [op.metadata.op_name for op in ops] == ["laminarCorrect"]


# --- its @build emits the NeoN nut / nuEff fields it owns ---
def test_build_emits_nut_and_nu_eff() -> None:
    names = [step.name for step in _runtime().run_build()]
    assert "fields.nut" in names
    assert "fields.nuEff" in names


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
