# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for the viscous-stress objects (``LinearViscousStress`` / ``OpenFOAMStress``).

Pure-Python: which stress a model *uses* is the model's own decision (tested in
``test_laminar_model`` / the fallback). Here we test the stress objects
themselves — their update operation and the OpenFOAM delegation. The numerical
assembly (``LinearViscousStress.divDevReff``) needs a mesh and is covered
end-to-end by ``test_laminar_comparison`` / ``test_pitzDaily_comparison``.
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


def test_stress_objects_have_one_update_operation() -> None:
    for stress in (LinearViscousStress(), OpenFOAMStress(_Model())):
        ops = stress.operations
        assert len(ops) == 1
        assert ops[0].operation_name == "update_viscous_stress"
