# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Turbulence framework (skeleton).

A minimal, extensible momentum-transport subsystem built on the NeoFOAM plugin
system. Native models register with :class:`momentumTransportModel`;
:func:`select_turbulence_model` resolves the configured model by name and falls
back to the OpenFOAM turbulence model (:class:`OpenFOAMTurbulenceModel`) when no
native model is registered.
"""

from .base import TurbulenceModel
from .fallback import OpenFOAMTurbulenceModel
from .momentumTransport import SpecMomentumTransport, momentumTransportModel
from .selection import model_name, select_from_case, select_turbulence_model

# Import side-effect: register the bundled native turbulence models.
from . import models  # noqa: E402

__all__ = [
    "momentumTransportModel",
    "SpecMomentumTransport",
    "TurbulenceModel",
    "OpenFOAMTurbulenceModel",
    "model_name",
    "select_turbulence_model",
    "select_from_case",
    "models",
]
