# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Subprocess helper: mesh one case (blockMesh → snappyHexMesh) and print stats.

Run as ``python _snappy_probe.py <case_dir>`` where ``<case_dir>`` holds
``system/{blockMeshDict,snappyHexMeshDict,controlDict,fvSchemes,fvSolution}`` and
``constant/triSurface/*.stl``. It builds the base mesh, refines/snaps it, validates
with ``checkMesh`` and prints one ``STATS <json>`` line, then ``os._exit``\\ s so the
GC teardown of the mesh-bound pybFoam objects (which SIGBUSes in-process) can't fail
the run. Mesh tools read their dicts / STLs relative to the working directory, so it
runs from inside the case.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import traceback

import pybFoam as pyf
from pybFoam.meshing import checkMesh, generate_blockmesh, generate_snappy_hex_mesh


def mesh_signature(case_dir: str) -> dict[str, object]:
    """Build blockMesh→snappy in ``case_dir`` and return the snapped mesh stats."""
    with contextlib.chdir(case_dir):
        time = pyf.Time(pyf.argList(["snappy", "-case", "."]))
        mesh = generate_blockmesh(
            time, pyf.dictionary.read("system/blockMeshDict"), verbose=False
        )
        generate_snappy_hex_mesh(
            mesh,
            pyf.dictionary.read("system/snappyHexMeshDict"),
            overwrite=True,
            verbose=False,
        )
        boundary = mesh.boundary()  # bind: iterating the temporary invalidates patches
        patches = sorted(
            [str(boundary[i].name()), boundary[i].size()] for i in range(len(boundary))
        )
        volumes = mesh.V()
        volume = round(sum(volumes[i] for i in range(len(volumes))), 9)
        passed = bool(checkMesh(mesh)["passed"])
        return {
            "nCells": mesh.nCells(),
            "nFaces": mesh.nFaces(),
            "nPoints": mesh.nPoints(),
            "patches": patches,
            "volume": volume,
            "checkMesh": passed,
        }


if __name__ == "__main__":
    try:
        print("STATS " + json.dumps(mesh_signature(sys.argv[1])))
        sys.stdout.flush()
        os._exit(0)
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(3)
