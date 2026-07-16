# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Parameter sweeps: describe a sweep once, export a runnable workflow, read it back.

The deep entry point is :class:`Sweep` — one value that owns the round-trip::

    from neofoam.tooling.workflow.sweep import Sweep

    sweep = Sweep(dimensions=dims, solver_name=solver, base_case=base, classes=cfg)
    sweep.export(out_dir)          # sweep.csv, params.yaml, Snakefile, sidecar, configs
    again = Sweep.load(out_dir)    # dims, rows, plan, snakefile all derived

``Sweep`` hides the cross-product, the Snakefile generation, the ``RulePlan``
resolution and the ``sweep.meta.json`` sidecar behind :meth:`~Sweep.export` /
:meth:`~Sweep.load` plus the read-only :attr:`~Sweep.rows` / :attr:`~Sweep.plan` /
:attr:`~Sweep.snakefile` views.

Interface (``__all__`` — grouped by role):

* **The round-trip** (what most callers need): :class:`Sweep`.
* **Canvas model** (the wizard's VueFlow node/edge layer): :class:`SweepDimension`,
  :func:`dim_node`, :func:`rule_nodes`, :func:`autowire`, :func:`nodes_to_dimensions`
  and the node/handle/edge-type constants (``ALL_RULE`` / ``SETUP_RULE`` /
  ``SOLVE_RULE`` / ``DIM_NODE_TYPE`` / ``RULE_NODE_TYPE`` / the ``*_PROPS`` /
  ``*_PREFIX`` constants).
* **Back-compat** (transitional — the free functions :class:`Sweep` composes, kept
  importable while callers migrate to the object): ``export_sweep`` / ``load_sweep`` /
  ``cross_product`` / ``sweep_snakefile`` / ``validate_dimensions`` /
  ``validate_mesh_dimension`` / ``validate_cad_dimension`` / ``variant_errors`` /
  ``CAD_RULE`` / :class:`LoadedSweep` / :class:`SweepExport`.

The implementation lives in the package-internal ``_canvas`` / ``_codegen`` / ``_io``
/ ``_validate`` modules; a new caller only needs :class:`Sweep` and the canvas model.
"""

from __future__ import annotations

from neofoam.tooling.workflow.sweep._canvas import (
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
from neofoam.tooling.workflow.sweep._codegen import sweep_snakefile
from neofoam.tooling.workflow.sweep._facade import Sweep
from neofoam.tooling.workflow.sweep._io import (
    CAD_RULE,
    LoadedSweep,
    SweepExport,
    cross_product,
    export_sweep,
    load_sweep,
)
from neofoam.tooling.workflow.sweep._validate import (
    validate_cad_dimension,
    validate_dimensions,
    validate_mesh_dimension,
    variant_errors,
)

__all__ = [
    # the deep entry point
    "Sweep",
    # canvas model (wizard node/edge layer)
    "SweepDimension",
    "dim_node",
    "rule_nodes",
    "autowire",
    "nodes_to_dimensions",
    "ALL_RULE",
    "SETUP_RULE",
    "SOLVE_RULE",
    "DIM_NODE_TYPE",
    "RULE_NODE_TYPE",
    "CFG_EDGE_PROPS",
    "FILE_EDGE_PROPS",
    "CFG_HANDLE_PREFIX",
    "IN_HANDLE_PREFIX",
    "OUT_HANDLE_PREFIX",
    # back-compat: the free functions Sweep composes
    "CAD_RULE",
    "LoadedSweep",
    "SweepExport",
    "cross_product",
    "export_sweep",
    "load_sweep",
    "sweep_snakefile",
    "validate_cad_dimension",
    "validate_dimensions",
    "validate_mesh_dimension",
    "variant_errors",
]
