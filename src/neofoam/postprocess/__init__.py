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
NeoFOAM, so a case file can select them by their ``type`` string. The nodes do
their arithmetic with the NeoN kernels, so a field is reduced on the executor it
lives on; pybFoam appears in the sources, the CSV writer and the reductions.
"""

from neofoam.postprocess.config import PostProcessConfig, TableSpec
from neofoam.postprocess.model import PostProcessor, postProcess
from neofoam.postprocess.node import (
    AggregatedData,
    AggregatedDataSet,
    BoundaryMesh,
    DataSets,
    FieldDataSets,
    InternalDataSet,
    InternalMesh,
    Node,
    PatchDataSet,
    Pipeline,
    PointDataSet,
    SamplingGeometry,
    SetGeometry,
    Source,
    SurfaceDataSet,
    SurfaceMesh,
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
    NeonCellGeometry,
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
from neofoam.postprocess.writers import CsvWriter, TableWriter, table_headers, table_rows

__all__ = [
    # Data contract
    "AggregatedData",
    "AggregatedDataSet",
    "InternalDataSet",
    "PatchDataSet",
    "SurfaceDataSet",
    "PointDataSet",
    "FieldDataSets",
    "DataSets",
    # Geometry protocols
    "InternalMesh",
    "BoundaryMesh",
    "SurfaceMesh",
    "SetGeometry",
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
    "NeonCellGeometry",
    "SurfaceGeometry",
    "PatchGeometry",
    "PointGeometry",
    # Output
    "TableWriter",
    "CsvWriter",
    "table_headers",
    "table_rows",
    # The framework model
    "postProcess",
    "PostProcessor",
]
