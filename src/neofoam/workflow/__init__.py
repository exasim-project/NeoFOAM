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

Where the STLs come from is out of scope — any upstream tool that writes
``constant/triSurface/*.stl`` plus a ``manifest.json`` plugs in here (the interactive
wizard's :mod:`neofoam.ui.geometry` derives the same facts straight from the STLs).
"""

from neofoam.workflow.patch_set import BoundingBox, PatchEntry, PatchSet, PatchRole
from neofoam.workflow.mesh_inputs import block_mesh_dict, snappy_dict

__all__ = [
    "BoundingBox",
    "PatchEntry",
    "PatchSet",
    "PatchRole",
    "block_mesh_dict",
    "snappy_dict",
]
