# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Subprocess helper: mesh one ``blockMeshDict`` and print its stats as JSON.

Run as ``python _blockmesh_probe.py <blockMeshDict>``; it builds the mesh with
``pybFoam.meshing.generate_blockmesh``, validates it with ``checkMesh`` and prints
one ``STATS <json>`` line to stdout. It ``os._exit``\\ s so the process never
reaches the GC teardown of the mesh-bound pybFoam objects (which SIGBUSes
in-process — a pre-existing pybFoam trait). The round-trip test (:mod:`test_block_mesh`)
invokes this once per dict so a teardown crash can't take down the pytest run, and
so each mesh is built in a clean OpenFOAM global state.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

import pybFoam as pyf
from pybFoam.meshing import checkMesh, generate_blockmesh

# Minimal system dicts: a Time + fvMesh need these present, but blockMesh itself
# only reads blockMeshDict. Stats depend solely on the mesh, not these values.
_CONTROL = (
    "FoamFile{version 2.0;format ascii;class dictionary;object controlDict;}\n"
    "application blockMesh;startFrom startTime;startTime 0;stopAt endTime;"
    "endTime 1;deltaT 1;writeControl timeStep;writeInterval 1;"
)
_SCHEMES = (
    "FoamFile{version 2.0;format ascii;class dictionary;object fvSchemes;}\n"
    "ddtSchemes{default steadyState;}gradSchemes{default Gauss linear;}"
    "divSchemes{default none;}laplacianSchemes{default Gauss linear corrected;}"
    "interpolationSchemes{default linear;}snGradSchemes{default corrected;}"
)
_SOLUTION = (
    "FoamFile{version 2.0;format ascii;class dictionary;object fvSolution;}\nsolvers{}"
)


def mesh_signature(dict_path: str) -> dict[str, object]:
    """Build the mesh from ``dict_path`` and return its invariant stats.

    The stats (cell/face/point counts, per-patch names+sizes, total volume) are a
    mesh fingerprint: identical stats ⇒ the two dicts produced the same mesh.
    """
    case = Path(tempfile.mkdtemp())
    (case / "system").mkdir()
    (case / "system" / "controlDict").write_text(_CONTROL)
    (case / "system" / "fvSchemes").write_text(_SCHEMES)
    (case / "system" / "fvSolution").write_text(_SOLUTION)

    time = pyf.Time(pyf.argList(["blockMesh", "-case", str(case)]))
    mesh = generate_blockmesh(time, pyf.dictionary.read(dict_path), verbose=False)

    boundary = mesh.boundary()  # bind: iterating the temporary invalidates patches
    patches = sorted(
        [str(boundary[i].name()), boundary[i].size()] for i in range(len(boundary))
    )
    volumes = mesh.V()
    volume = round(sum(volumes[i] for i in range(len(volumes))), 10)
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
    except BaseException:  # surface a real error before the hard exit
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(3)
