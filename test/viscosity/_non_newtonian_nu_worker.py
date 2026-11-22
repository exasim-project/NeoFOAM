# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Builds the OpenFOAM viscosity fallback on a real non-Newtonian case.

Run as ``python _non_newtonian_nu_worker.py <case_dir>``; writes
``<case_dir>/nu.json``. A subprocess because one process owns exactly one
``Foam::Time``, and the fallback needs a live mesh-bound
``singlePhaseTransportModel`` — the object whose ``nu()`` binding is what
:mod:`test_non_newtonian_fallback_nu` is about.

The four calls under ``run`` are exactly the ones
``incompressibleFluid.create_fields`` makes (``create_laminar_transport``,
``select_viscosity_model``, ``build_viscosity``, ``create_nu``).
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
from pybFoam import volVectorField
from pybFoam.turbulence import singlePhaseTransportModel

from neofoam.viscosity.config import TransportPropertiesConfig
from neofoam.viscosity.selection import select_viscosity_model


def _generate_mesh(case_dir: Path, runtime: Any) -> None:
    """Generate constant/polyMesh from system/blockMeshDict, in process."""
    block_dict = pyf.dictionary.read(str(case_dir / "system" / "blockMeshDict"))
    generated = pyf.meshing.generate_blockmesh(runtime, block_dict)
    del generated  # drop the registered region0 mesh before reading it back
    gc.collect()


def run(case_dir: Path) -> dict[str, Any]:
    """Select and build the viscosity model the way the solver does; return its ``nu``."""
    arg_list = pyf.argList(["nonNewtonianNuWorker"])
    runtime = pyf.Time(arg_list)
    _generate_mesh(case_dir, runtime)
    mesh = pyf.fvMesh(runtime)

    U = volVectorField.read_field(mesh, "U")
    phi = pyf.createPhi(U)
    transport = singlePhaseTransportModel(U, phi)

    config = TransportPropertiesConfig.load(case_dir=case_dir)
    selected = select_viscosity_model(config)
    nu = selected.build(transport=transport).nu_field()

    return {
        "name": str(nu.name()),
        "selected": type(selected).__name__,
        "internal": np.asarray(nu.internalField()).tolist(),
    }


def main() -> None:
    case_dir = Path(sys.argv[1]).resolve()
    os.chdir(case_dir)
    (case_dir / "nu.json").write_text(json.dumps(run(case_dir)))


if __name__ == "__main__":
    main()
