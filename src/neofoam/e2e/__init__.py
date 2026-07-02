# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""End-to-end agentic CFD workflow: CAD geometry -> mesh -> solve -> report.

See ``specs/e2e_agentic_workflow.md`` for the design. The package is staged:

* stage 1 -- :mod:`neofoam.e2e.extract` extracts named STL patches from a CAD
  model and emits a :class:`~neofoam.e2e.manifest.PatchManifest` (the contract
  that ties geometry -> mesh -> boundary conditions together).
"""

from neofoam.e2e.config import MissingConfigError, require_configs
from neofoam.e2e.geometry import stage_geometry
from neofoam.e2e.manifest import BoundingBox, PatchEntry, PatchManifest, PatchRole
from neofoam.e2e.mesh_inputs import block_mesh_dict, snappy_dict

__all__ = [
    "BoundingBox",
    "MissingConfigError",
    "PatchEntry",
    "PatchManifest",
    "PatchRole",
    "block_mesh_dict",
    "require_configs",
    "snappy_dict",
    "stage_geometry",
]
