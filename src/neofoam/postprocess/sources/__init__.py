# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The sources shipped with NeoFOAM: cell fields, patches, probes, surfaces, residuals.

A source is where a table reaches into the simulation backend, so this is where
the geometry adapters live too. Importing this subpackage registers every source
(and the :class:`~neofoam.postprocess.sources.sampling.Sample` node that goes
with the surfaces), so a case file can select them by their ``type`` string.
"""

from neofoam.postprocess.sources.fields import (
    InternalField,
    LineField,
    PatchField,
    field,
    line,
    patch,
)
from neofoam.postprocess.sources.geometry import (
    CellGeometry,
    NeonCellGeometry,
    PatchGeometry,
    PointGeometry,
    SurfaceGeometry,
)
from neofoam.postprocess.sources.residuals import Residuals, residuals
from neofoam.postprocess.sources.sampling import (
    IsoSurface,
    PlaneSurface,
    Sample,
    iso_surface,
    plane,
)

__all__ = [
    # Field sources
    "InternalField",
    "PatchField",
    "LineField",
    "field",
    "patch",
    "line",
    # Sampled surfaces
    "PlaneSurface",
    "IsoSurface",
    "plane",
    "iso_surface",
    "Sample",
    # Solver residuals
    "Residuals",
    "residuals",
    # Geometry adapters
    "CellGeometry",
    "NeonCellGeometry",
    "SurfaceGeometry",
    "PatchGeometry",
    "PointGeometry",
]
