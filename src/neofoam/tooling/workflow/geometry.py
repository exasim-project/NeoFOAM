# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Geometry → mesh: the manifest schema and the deterministic mesh-dict mappers.

The one entry point for going from staged geometry to mesh inputs. Whatever stages
a case's geometry writes ``<case>/manifest.json`` — a :class:`PatchSet` — and this
module both *is* that schema and knows how to turn it into the OpenFOAM mesh dicts.

The common case is one call::

    from neofoam.tooling.workflow.geometry import PatchSet, build_mesh_inputs

    patch_set = PatchSet.load("manifest.json")
    block, snappy, preprocess = build_mesh_inputs(patch_set)  # snappy is None for a box

Interface (``__all__``):

* **Manifest schema** (the geometry contract): :class:`PatchSet`, :class:`PatchEntry`,
  :class:`PatchRole`, :class:`BoundingBox`.
* **Mesh mappers** (deterministic, no OpenFOAM, no LLM): :func:`build_mesh_inputs`
  (the headline verb — both dicts + the preprocess enable-list from a manifest) and
  the two lower-level pieces it composes, :func:`block_mesh_dict` / :func:`snappy_dict`.

The schema lives in :mod:`neofoam.tooling.workflow.patch_set` and the mappers in
:mod:`neofoam.tooling.workflow.mesh_inputs`; import them from **here**, not those
modules, so their internal layout can change without breaking callers.
"""

from __future__ import annotations

from neofoam.tooling.workflow.mesh_inputs import (
    block_mesh_dict,
    build_mesh_inputs,
    snappy_dict,
)
from neofoam.tooling.workflow.patch_set import (
    BoundingBox,
    PatchEntry,
    PatchRole,
    PatchSet,
)

__all__ = [
    "PatchSet",
    "PatchEntry",
    "PatchRole",
    "BoundingBox",
    "build_mesh_inputs",
    "block_mesh_dict",
    "snappy_dict",
]
