# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Turbulence (momentum-transport) subsystem.

One plugin family — :class:`momentumTransportModel` — holds every turbulence
model as a :class:`~neofoam.framework.model.ModelSpec`.
:func:`select_turbulence_model` resolves the configured model by name and, via a
single ``fallback`` flag, builds either the native-NeoN handle
(:class:`~neofoam.turbulence.native.NeoNHandle`, ``fallback=False``) or the
pybFoam-OpenFOAM handle
(:class:`~neofoam.turbulence.fallback.FallbackHandle`, ``fallback=True``). Both
satisfy the :class:`~neofoam.turbulence.protocol.MomentumTransport` surface.

Importing this package registers the bundled models, whose closures author NeoN
field maths — so it requires the NeoN bindings (the same monorepo build that
provides pybFoam).
"""

from .base import ViscousStress
from .fallback import FallbackHandle, OpenFOAMTurbulenceModel
from .momentumTransport import momentumTransportModel
from .native import NeoNHandle
from .protocol import MomentumTransport
from .selection import model_name, select_from_case, select_turbulence_model
from .stress import LinearViscousStress, OpenFOAMStress

# Import side-effect: register the bundled turbulence models.
from . import models  # noqa: E402

__all__ = [
    "momentumTransportModel",
    "MomentumTransport",
    "NeoNHandle",
    "FallbackHandle",
    "ViscousStress",
    "LinearViscousStress",
    "OpenFOAMStress",
    "OpenFOAMTurbulenceModel",
    "model_name",
    "select_turbulence_model",
    "select_from_case",
    "models",
]
