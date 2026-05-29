# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Turbulence framework (skeleton).

A minimal, extensible turbulence subsystem built on the NeoFOAM plugin system.
Native models register with :class:`turbulenceModel`; :func:`select_turbulence_model`
resolves the configured model by name and falls back to the OpenFOAM turbulence
model (:class:`OpenFOAMTurbulenceModel`) when no native model is registered.

``TurbulencePropertiesConfig`` is intentionally NOT re-exported here so that
importing :mod:`neofoam.turbulence` stays free of ``neofoam.io`` / pybFoam;
import it explicitly from :mod:`neofoam.turbulence.config` when needed.
"""

from .base import TurbulenceModel
from .fallback import OpenFOAMTurbulenceModel
from .interface import turbulenceModel
from .selection import model_name, select_from_case, select_turbulence_model

# Import side-effect: register the bundled native turbulence models.
from . import models  # noqa: E402

__all__ = [
    "turbulenceModel",
    "TurbulenceModel",
    "OpenFOAMTurbulenceModel",
    "model_name",
    "select_turbulence_model",
    "select_from_case",
    "models",
]
