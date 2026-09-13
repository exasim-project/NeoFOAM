# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The nodes shipped with NeoFOAM: selectors, binners, field functions, aggregators, output.

Importing this subpackage registers every one of them, so a case file can select
them by their ``type`` string. Nothing here imports pybFoam — a node sees the
mesh only through the :class:`~neofoam.postprocess.node.Geometry` protocol.
"""

from neofoam.postprocess.nodes.aggregators import (
    GREAT,
    Max,
    Mean,
    Min,
    Sum,
    SurfIntegrate,
    VolIntegrate,
    group_sum,
)
from neofoam.postprocess.nodes.binning import Directional
from neofoam.postprocess.nodes.debug import Print, Scale
from neofoam.postprocess.nodes.field_functions import Area, Component, Mag
from neofoam.postprocess.nodes.rows import Rows
from neofoam.postprocess.nodes.selectors import Binary, Box, Not, Selector, Sphere

__all__ = [
    # Selectors
    "Selector",
    "Box",
    "Sphere",
    "Not",
    "Binary",
    # Binning
    "Directional",
    # Field functions
    "Mag",
    "Component",
    "Area",
    # Aggregators
    "Sum",
    "Mean",
    "Max",
    "Min",
    "VolIntegrate",
    "SurfIntegrate",
    "GREAT",
    "group_sum",
    # Per-element output
    "Rows",
    # Debug
    "Print",
    "Scale",
]
