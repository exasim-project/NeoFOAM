# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""One-role-per-process worker for the NeoN local-patch-type translation test.

Exactly one ``Foam::Time`` per process, so meshing and the field read each run in
their own process and hand their result to the parent through a file.

Roles (``python _local_patch_types_worker.py <role> <case_dir>``):

* ``mesh``   — generate the block mesh.
* ``fields`` — read ``0/U`` and ``0/p`` through the NeoN field reader and save the
  boundary values, split per patch, to ``U_boundary.npz`` / ``p_boundary.npz``.
  An unsupported patch type aborts here instead.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import neon._neon as nn
import numpy as np
import pybFoam as pyf

from neofoam import neofoam_bindings as nfb

os.environ.setdefault("FOAM_SIGFPE", "false")


def role_mesh(case: Path) -> None:
    """Generate the block mesh from ``system/blockMeshDict``."""
    time = pyf.Time(str(case.parent), case.name)
    block_dict = pyf.dictionary.read(str(case / "system" / "blockMeshDict"))
    pyf.meshing.generate_blockmesh(time, block_dict, False, "constant")


def _per_patch(runtime: Any, values: np.ndarray) -> dict[str, np.ndarray]:
    """Split flat boundary values into named patches, in OpenFOAM patch order."""
    # The boundary mesh is indexed, not iterated: pybFoam's fvBoundaryMesh.__iter__
    # aborts the process.
    boundary = runtime.mesh.boundary()
    per_patch = {}
    start = 0
    for patch_id in range(len(boundary)):
        patch = boundary[patch_id]
        end = start + patch.size()
        per_patch[str(patch.name())] = values[start:end]
        start = end
    return per_patch


def role_fields(case: Path) -> None:
    """Read U and p through the NeoN reader → one ``.npz`` of patch values each."""
    nn.initialize(["neon"])

    # Foam::Time keeps a raw reference to the argList and the NeoN runtime a raw
    # reference to the Time, so all three stay bound for the lifetime of the read.
    arg_list = pyf.argList(["local_patch_types", "-case", str(case)])
    neon_time = pyf.Time(arg_list)
    rt = nfb.create_adapter_run_time(neon_time)
    rt.fv_schemes_dict = nfb.map_fv_schemes(rt.fv_schemes_dict)

    # NeoN exposes the boundary values of a scalar field itself; the vector field's go
    # through NeoFOAM's `vector_boundary_values`.
    velocity = nfb.vector_boundary_values(nfb.read_vector_volume_field(rt, "U"))
    pressure = nfb.read_scalar_volume_field(rt, "p").boundary_data_value()
    for name, values in (("U", velocity), ("p", pressure)):
        host = np.asarray(values.copy_to_host())
        np.savez(case / f"{name}_boundary.npz", **_per_patch(rt, host))


_ROLES = {"mesh": role_mesh, "fields": role_fields}


def main() -> None:
    role, case_dir = sys.argv[1], Path(sys.argv[2])
    _ROLES[role](case_dir)
    # Result is already on disk; NeoN/Kokkos + OpenFOAM teardown at interpreter exit
    # is fragile, so exit hard.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
