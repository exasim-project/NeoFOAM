# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Batch alpha-Courant evaluator for one staged synthetic case (one ``Foam::Time``).

Run as ``python _alpha_courant_worker.py <case_dir> <request.json>``; writes
``<case_dir>/result.json``. One process owns exactly one ``Foam::Time`` and one
mesh, so every scenario of a mesh is evaluated here in a single pass: the mesh
is generated in-process from ``system/blockMeshDict``, ``alpha.water``/``U`` are
re-seeded per scenario, and ``compute_alpha_courant_number`` is called on the
resulting state.

The request is ``{"scenarios": {name: {"alpha": [per-cell values],
"U": [x, y, z]}}}`` — both are spelled out cell by cell in the test so that
every expected number stays hand-derivable.
"""

from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pybFoam as pyf
from pybFoam import fvc, volScalarField, volVectorField

from neofoam.solver.incompressibleVoF.incompressibleVoF import (
    compute_alpha_courant_number,
)


def _generate_mesh(case_dir: Path, runtime: Any) -> None:
    """Generate constant/polyMesh from system/blockMeshDict, in process."""
    block_dict = pyf.dictionary.read(str(case_dir / "system" / "blockMeshDict"))
    generated = pyf.meshing.generate_blockmesh(runtime, block_dict)
    del generated  # drop the registered region0 mesh before reading it back
    gc.collect()


def run(case_dir: Path, request: dict[str, Any]) -> dict[str, list[float]]:
    """Evaluate every scenario on this case's mesh; keep the frame alive."""
    arg_list = pyf.argList(["alphaCourantWorker"])
    runtime = pyf.Time(arg_list)
    _generate_mesh(case_dir, runtime)
    mesh = pyf.fvMesh(runtime)

    u = volVectorField.read_field(mesh, "U")
    phi = pyf.createPhi(u)
    alpha1 = volScalarField.read_field(mesh, "alpha.water")
    n_cells = np.asarray(alpha1.internalField()).shape[0]

    results: dict[str, list[float]] = {}
    for name, spec in request["scenarios"].items():
        u_view = np.asarray(u.internalField())
        u_view[:] = np.tile(np.asarray(spec["U"], dtype=float), (n_cells, 1))
        u.correctBoundaryConditions()
        phi.assign(fvc.flux(u))

        alpha_view = np.asarray(alpha1.internalField())
        alpha_view[:] = np.asarray(spec["alpha"], dtype=float)
        alpha1.correctBoundaryConditions()

        results[name] = list(compute_alpha_courant_number(phi, alpha1))
    return results


def main() -> None:
    case_dir = Path(sys.argv[1]).resolve()
    request = json.loads(Path(sys.argv[2]).read_text())
    os.chdir(case_dir)
    (case_dir / "result.json").write_text(json.dumps(run(case_dir, request)))


if __name__ == "__main__":
    main()
