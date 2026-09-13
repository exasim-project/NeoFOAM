# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Post-processing: declare a table as a source plus nodes, write it as CSV.

The data contract (:mod:`~neofoam.postprocess.node`) and the case's front doors
stay here — the loaders behind those doors
(:func:`~neofoam.postprocess.table.tables_for_case` and what it calls) are the
``postProcess`` model's, and are imported from their own modules. The plugins
are grouped into :mod:`~neofoam.postprocess.nodes`,
:mod:`~neofoam.postprocess.sources` and :mod:`~neofoam.postprocess.writers`.
Importing this package registers every node, source and writer shipped with
NeoFOAM, so a case file can select them by their ``type`` string. The nodes are
pure numpy; pybFoam appears in the sources, the CSV writer and the reductions.
"""

from neofoam.postprocess.config import PostProcessConfig, TableSpec
from neofoam.postprocess.model import PostProcessor, postProcess
from neofoam.postprocess.node import (
    AggregatedDataSet,
    DataSet,
    Geometry,
    Node,
    Pipeline,
    SamplingGeometry,
    Source,
)
from neofoam.postprocess.nodes import (
    GREAT,
    Area,
    Binary,
    Box,
    Component,
    Directional,
    Mag,
    Max,
    Mean,
    Min,
    Not,
    Print,
    Rows,
    Scale,
    Selector,
    Sphere,
    Sum,
    SurfIntegrate,
    VolIntegrate,
    group_sum,
)
from neofoam.postprocess.sources import (
    CellGeometry,
    IsoSurface,
    LineField,
    PatchField,
    PatchGeometry,
    PlaneSurface,
    PointGeometry,
    Residuals,
    Sample,
    SurfaceGeometry,
    field,
    iso_surface,
    line,
    patch,
    plane,
    residuals,
)
from neofoam.postprocess.table import Table, TableSet
from neofoam.postprocess.writers import CsvWriter, TableWriter

__all__ = [
    # Data contract
    "AggregatedDataSet",
    "DataSet",
    "Geometry",
    "SamplingGeometry",
    # Plugin families
    "Node",
    "Source",
    # Pipelines
    "Pipeline",
    "field",
    # Sources
    "PatchField",
    "LineField",
    "Residuals",
    "patch",
    "line",
    "residuals",
    # Nodes - selectors
    "Selector",
    "Box",
    "Sphere",
    "Not",
    "Binary",
    # Nodes - binning
    "Directional",
    # Nodes - field functions
    "Mag",
    "Component",
    "Area",
    # Nodes - aggregators
    "Sum",
    "Mean",
    "Max",
    "Min",
    "VolIntegrate",
    "SurfIntegrate",
    "GREAT",
    "group_sum",
    # Nodes - per-element output
    "Rows",
    # Sources - sampled surfaces
    "PlaneSurface",
    "IsoSurface",
    "plane",
    "iso_surface",
    "Sample",
    # Nodes - debug
    "Print",
    "Scale",
    # Tables and the case's two front doors
    "Table",
    "TableSet",
    "TableSpec",
    "PostProcessConfig",
    # Geometry adapters
    "CellGeometry",
    "SurfaceGeometry",
    "PatchGeometry",
    "PointGeometry",
    # Output
    "TableWriter",
    "CsvWriter",
    # The framework model
    "postProcess",
    "PostProcessor",
]
