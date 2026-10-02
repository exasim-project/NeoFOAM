# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for the ``laminar`` momentum-transport model's native NeoN shape.

After the family merge ``laminar`` is a dual-shape model in the single
``momentumTransportModel`` family:

* native NeoN (``fallback=False``): its ``@build`` emits ``nut = 0`` and the
  surface ``nuEff``; it solves no transport equation, so it declares **no native
  ``@operation``** — the two claims below.
* pybFoam fallback (``fallback=True``): one co-located ``fallback=True``
  ``laminarCorrect`` op, pinned (with every other model's) by
  ``test_selection.test_registered_model_schedules_its_own_fallback_correct_op``.

Structure is checked without a live NeoN runtime; the end-to-end NeoN ``nut``/
``nuEff`` values are covered bit-for-bit by ``test_neon_turbulence_parity``, and the
case's own config by ``test_config``.
"""

from typing import Any

from neofoam.turbulence.models.laminar import laminar
from turbulence.conftest import case_for

#: Point the laminar model at the case the solver would feed it.
LAMINAR = case_for("laminar")


def _runtime() -> Any:
    return laminar.instantiate(LAMINAR.path)


def test_declares_no_native_operation() -> None:
    assert _runtime().native_operations() == []


def test_build_emits_nut_and_nu_eff() -> None:
    names = [step.name for step in _runtime().run_build()]
    assert "fields.nut" in names
    assert "fields.nuEff" in names
