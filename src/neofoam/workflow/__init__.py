# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Case-staging workflow: the geometry contract read by the MCP tools.

:mod:`neofoam.workflow.patch_set` — the :class:`~neofoam.workflow.patch_set.PatchSet`
schema (``<case>/manifest.json``): named boundary patches with CFD roles, the
bounding box and mesh sizing facts. It ties geometry → mesh → boundary conditions
together; :func:`neofoam.mcp.tools.case_patches` reads it so an agent never invents
patch names or roles.

The mesh-mapping and parameter-sweep halves of this package (``mesh_inputs``,
``sweep``, ``paramspace``, ``rules``, ``snakemake_dag``) ship separately; this module
intentionally re-exports only the geometry contract so importing it pulls no
``neofoam.tools`` geometry machinery.
"""

from neofoam.workflow.patch_set import (
    BoundingBox,
    BoxFace,
    PatchEntry,
    PatchRole,
    PatchSet,
)

__all__ = [
    "BoundingBox",
    "BoxFace",
    "PatchEntry",
    "PatchRole",
    "PatchSet",
]
