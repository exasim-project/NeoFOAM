# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Viscosity (transport) framework (skeleton).

A minimal, extensible viscosity subsystem built on the NeoFOAM plugin system,
mirroring :mod:`neofoam.turbulence`. Native models register with
:class:`viscosityModel`; :func:`select_viscosity_model` resolves the configured
``transportModel`` by name and falls back to pybFoam's
``singlePhaseTransportModel`` (:class:`OpenFOAMViscosityModel`) when no native
model is registered.

``TransportPropertiesConfig`` is intentionally NOT re-exported here so that
importing :mod:`neofoam.viscosity` stays free of ``neofoam.io`` / pybFoam;
import it explicitly from :mod:`neofoam.viscosity.config` when needed.
"""

from .base import ViscosityModel
from .fallback import OpenFOAMViscosityModel
from .interface import viscosityModel
from .selection import model_name, select_from_case, select_viscosity_model

# Import side-effect: register the bundled native viscosity models.
from . import models  # noqa: E402

__all__ = [
    "viscosityModel",
    "ViscosityModel",
    "OpenFOAMViscosityModel",
    "model_name",
    "select_viscosity_model",
    "select_from_case",
    "models",
]
