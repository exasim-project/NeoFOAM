# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""UI-free data layer for the case wizard's parameter-sweep canvas (facade).

The sweep is split by responsibility across four modules; this module re-exports
their public surface so ``neofoam.tooling.workflow.sweep`` stays the one import
path callers use:

- :mod:`neofoam.tooling.workflow.sweep_canvas` — the VueFlow node/edge model
  (``dim_node``/``rule_nodes``/``autowire``/``nodes_to_dimensions`` + constants).
- :mod:`neofoam.tooling.workflow.sweep_validate` — per-variant validation
  (``validate_dimensions``/``validate_mesh_dimension``/``validate_cad_dimension``/
  ``variant_errors``).
- :mod:`neofoam.tooling.workflow.sweep_codegen` — Snakefile generation
  (``sweep_snakefile``).
- :mod:`neofoam.tooling.workflow.sweep_io` — export to / load from a runnable
  workflow directory (``export_sweep``/``load_sweep`` + ``cross_product``).

The maps: a *dimension* node is a live view into one ``params.yaml`` section (one
solver config holding named variants); the reserved ``mesh`` dimension is keyed
(its variants carry config-name-keyed payloads and the mesh rules run once per
variant under ``meshes/{variant}/``, shared by every case). The pipeline comes
from the packaged rule library (:mod:`neofoam.tooling.workflow.rules`).
"""

from __future__ import annotations

from neofoam.tooling.workflow.sweep_canvas import (
    ALL_RULE,
    CFG_EDGE_PROPS,
    CFG_HANDLE_PREFIX,
    DIM_NODE_TYPE,
    FILE_EDGE_PROPS,
    IN_HANDLE_PREFIX,
    OUT_HANDLE_PREFIX,
    RULE_NODE_TYPE,
    SETUP_RULE,
    SOLVE_RULE,
    SweepDimension,
    autowire,
    dim_node,
    nodes_to_dimensions,
    rule_nodes,
)
from neofoam.tooling.workflow.sweep_codegen import sweep_snakefile
from neofoam.tooling.workflow.sweep_io import (
    CAD_RULE,
    LoadedSweep,
    SweepExport,
    cross_product,
    export_sweep,
    load_sweep,
)
from neofoam.tooling.workflow.sweep_validate import (
    validate_cad_dimension,
    validate_dimensions,
    validate_mesh_dimension,
    variant_errors,
)

__all__ = [
    "ALL_RULE",
    "CAD_RULE",
    "CFG_EDGE_PROPS",
    "CFG_HANDLE_PREFIX",
    "DIM_NODE_TYPE",
    "FILE_EDGE_PROPS",
    "IN_HANDLE_PREFIX",
    "LoadedSweep",
    "OUT_HANDLE_PREFIX",
    "RULE_NODE_TYPE",
    "SETUP_RULE",
    "SOLVE_RULE",
    "SweepDimension",
    "SweepExport",
    "autowire",
    "cross_product",
    "dim_node",
    "export_sweep",
    "load_sweep",
    "nodes_to_dimensions",
    "rule_nodes",
    "sweep_snakefile",
    "validate_cad_dimension",
    "validate_dimensions",
    "validate_mesh_dimension",
    "variant_errors",
]
