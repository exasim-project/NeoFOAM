# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Viscosity field builders.

Helpers to build the kinematic-viscosity ``dimensionedScalar`` fields the
viscosity and momentum-transport models register in the Context (``nu``, ``nut``).
pybFoam is imported lazily, so importing this module needs no built OpenFOAM
environment.
"""

from typing import Any

__all__ = ["VISCOSITY_DIMENSIONS", "dimensioned_viscosity"]

#: Kinematic-viscosity dimensions [0 2 -1 0 0 0 0].
VISCOSITY_DIMENSIONS = (0.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0)


def dimensioned_viscosity(name: str, value: float) -> Any:
    """Build a named kinematic-viscosity ``dimensionedScalar`` from a value.

    The pybFoam transport binding exposes no ``nu()``; native models build the
    field from the value OpenFOAM reads from ``constant/transportProperties``.
    """
    import pybFoam

    return pybFoam.dimensionedScalar(
        pybFoam.Word(name), pybFoam.dimensionSet(*VISCOSITY_DIMENSIONS), value
    )
