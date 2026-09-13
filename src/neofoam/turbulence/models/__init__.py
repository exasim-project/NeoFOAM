# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Bundled momentum-transport models.

Importing this package registers every bundled model with the single
:class:`~neofoam.turbulence.momentumTransport.momentumTransportModel` family
(registration is an import side-effect of each model module). Each model is a
native-NeoN closure that also declares a co-located ``fallback=True`` ``correct``
op, so the same model serves both ``incompressibleFluidNeoN`` (native) and
``incompressibleFluid`` (pybFoam fallback).

Because these closures author on-device NeoN field maths, importing this package
(and therefore ``neofoam.turbulence``) requires the NeoN bindings to be built —
the same monorepo build that provides pybFoam.
"""

from .kEpsilon import kEpsilon
from .kOmegaSST import kOmegaSST
from .laminar import laminar
from .realizableKE import realizableKE
from .spalartAllmaras import spalartAllmaras

__all__ = ["laminar", "kEpsilon", "kOmegaSST", "spalartAllmaras", "realizableKE"]
