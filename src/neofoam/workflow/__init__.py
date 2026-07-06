# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Case-staging workflow: the geometry contract and the deterministic mesh mappers.

* :mod:`neofoam.workflow.patch_set` — the :class:`~neofoam.workflow.patch_set.PatchSet`
  schema (``<case>/manifest.json``): named boundary patches with CFD roles, the
  bounding box and mesh sizing facts. It ties geometry → mesh → boundary conditions
  together; :func:`neofoam.mcp.tools.case_patches` reads it so an agent never invents
  patch names or roles.
* :mod:`neofoam.workflow.mesh_inputs` maps a ``PatchSet`` to the ``blockMeshDict`` /
  ``snappyHexMeshDict`` configs (deterministic, no OpenFOAM, no LLM).
* :mod:`neofoam.workflow.sweep` + :mod:`neofoam.workflow.paramspace` — parameter
  sweeps over a saved base case: named config variants (``params.yaml``), their
  cross product (``sweep.csv``) and a generated Snakemake workflow composed from
  the packaged rule library (:mod:`neofoam.workflow.rules`): per mesh variant
  ``setup_mesh`` → blockMesh → snappyHexMesh → checkMesh (built ONCE under
  ``meshes/{variant}/``), then per case ``setup`` (clone base + apply configs +
  copy the variant's mesh, via :mod:`neofoam.workflow.sweep_runner`) → ``solve``.
  The wizard's Parameters step drives this canvas.
* :mod:`neofoam.workflow.snakemake_dag` — renders an exported sweep's
  ``snakemake --dag`` / ``--rulegraph`` as canvas nodes and edges (the DAG view
  below the wizard's Parameters canvas).

Where the STLs come from is out of scope — any upstream tool that writes
``constant/triSurface/*.stl`` plus a ``manifest.json`` plugs in here (the interactive
wizard's :mod:`neofoam.ui.geometry` derives the same facts straight from the STLs).
"""

from neofoam.workflow.patch_set import BoundingBox, PatchEntry, PatchSet, PatchRole
from neofoam.workflow.mesh_inputs import block_mesh_dict, snappy_dict
from neofoam.workflow.paramspace import KeyedDim, YamlParamSpace
from neofoam.workflow.rules import (
    RulePlan,
    RuleRegistry,
    RuleSpec,
    default_registry,
    rules_dir,
)
from neofoam.workflow.snakemake_dag import dag_graph
from neofoam.workflow.sweep import (
    LoadedSweep,
    SweepDimension,
    cross_product,
    export_sweep,
    load_sweep,
    sweep_snakefile,
)

__all__ = [
    "BoundingBox",
    "KeyedDim",
    "LoadedSweep",
    "PatchEntry",
    "PatchSet",
    "PatchRole",
    "RulePlan",
    "RuleRegistry",
    "RuleSpec",
    "SweepDimension",
    "YamlParamSpace",
    "block_mesh_dict",
    "cross_product",
    "dag_graph",
    "default_registry",
    "export_sweep",
    "load_sweep",
    "rules_dir",
    "snappy_dict",
    "sweep_snakefile",
]
