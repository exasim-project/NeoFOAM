# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for the viscous-stress objects (``LinearViscousStress`` / ``OpenFOAMStress``).

Pure-Python: which stress a model *uses* is the model's own decision, and the
*operation* that drives the stress is owned by the model (tested in
``test_laminar_model`` / the fallback). Here we test the stress objects
themselves — the OpenFOAM delegation and that ``update`` is a no-op for the
fallback. The numerical assembly (``LinearViscousStress.divDevReff``) needs a mesh
and is covered end-to-end by ``test_laminar_comparison`` / ``test_pitzDaily_comparison``.
"""

from typing import Any

from neofoam.turbulence.stress import LinearViscousStress, OpenFOAMStress


class _Model:
    """Minimal stand-in for a momentum-transport model OpenFOAMStress delegates to."""

    def divDevReff(self, U: Any) -> Any:
        return ("delegated", U)


def test_openfoam_stress_delegates_to_the_model() -> None:
    stress = OpenFOAMStress(_Model())
    assert stress.divDevReff("U") == ("delegated", "U")


def test_openfoam_stress_update_is_a_noop() -> None:
    # The OpenFOAM model owns its eddy viscosity, so the stress has nothing to
    # refresh from the Context — update is a side-effect-free no-op.
    OpenFOAMStress(_Model()).update(ctx=object())


def test_stress_objects_have_no_operations_attribute() -> None:
    # The update operation now lives on the momentum-transport model, not here.
    for stress in (LinearViscousStress(), OpenFOAMStress(_Model())):
        assert not hasattr(stress, "operations")
