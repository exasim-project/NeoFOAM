# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""One-role-per-process worker for the kEpsilon near-wall-distance test.

Same constraint as :mod:`_parity_worker`: exactly one ``Foam::Time`` per process,
so mesh generation and the NeoN model build each run in their own process and
hand their result to the parent through a file.

Roles (``python _near_wall_dist_worker.py <role> <case_dir>``):

* ``mesh``          — generate the block mesh.
* ``near_wall_dist`` — build the native NeoN kEpsilon closure on that case and
  save its near-wall distance, split per boundary patch, to ``near_wall_dist.npz``.
  Reaching the closure's build at all is the regression: the case's ``fvSchemes``
  carries no ``wallDist`` block.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import neon._neon as nn
import numpy as np
import pybFoam as pyf

from neofoam import neofoam_bindings as nfb
from neofoam.turbulence.config import TurbulencePropertiesConfig
from neofoam.turbulence.selection import select_turbulence_model

os.environ.setdefault("FOAM_SIGFPE", "false")


def role_mesh(case: Path) -> None:
    """Generate the block mesh from ``system/blockMeshDict``."""
    time = pyf.Time(str(case.parent), case.name)
    block_dict = pyf.dictionary.read(str(case / "system" / "blockMeshDict"))
    pyf.meshing.generate_blockmesh(time, block_dict, False, "constant")


def role_near_wall_dist(case: Path) -> None:
    """Build the NeoN kEpsilon closure → ``near_wall_dist.npz`` (one array per patch)."""
    nn.initialize(["neon"])

    cfg = TurbulencePropertiesConfig.load(case_dir=str(case))
    # Foam::Time keeps a raw reference to the argList and the NeoN runtime a raw
    # reference to the Time, so all three stay bound (see _parity_worker.role_subject).
    arg_list = pyf.argList(["near_wall_dist", "-case", str(case)])
    neon_time = pyf.Time(arg_list)
    rt = nfb.create_adapter_run_time(neon_time)
    rt.fv_schemes_dict = nfb.map_fv_schemes(rt.fv_schemes_dict)

    U = nfb.read_vector_volume_field(rt, "U")
    nu = nfb.create_uniform_volume_field(rt, "nu", nfb.read_transport_viscosity(rt))
    turbulence = select_turbulence_model(cfg, fallback=False, runtime=rt, nu=nu, case_dir=case)
    turbulence.validate(U)

    # The closure's own near-wall distance model — the exact object its epsilon /
    # nutk wall functions read through the BoundaryContext. Reached through the
    # handle's Context because the handle exposes fields, not models.
    near_wall_dist = turbulence._ctx.models["kEpsilon_nearWallDist"]
    values = np.asarray(near_wall_dist.boundary_data_value().copy_to_host())

    # Boundary values are stored flat in patch order (what constructFrom writes), so
    # the OpenFOAM patch sizes split them back into named patches. The boundary mesh
    # is indexed, not iterated: pybFoam's fvBoundaryMesh.__iter__ aborts the process.
    boundary = rt.mesh.boundary()
    per_patch = {}
    start = 0
    for patch_id in range(len(boundary)):
        patch = boundary[patch_id]
        end = start + patch.size()
        per_patch[str(patch.name())] = values[start:end]
        start = end
    np.savez(case / "near_wall_dist.npz", **per_patch)


_ROLES = {"mesh": role_mesh, "near_wall_dist": role_near_wall_dist}


def main() -> None:
    role, case_dir = sys.argv[1], Path(sys.argv[2])
    _ROLES[role](case_dir)
    # Result is already on disk; NeoN/Kokkos + OpenFOAM teardown at interpreter exit
    # is fragile, so exit hard (see _parity_worker.main).
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
