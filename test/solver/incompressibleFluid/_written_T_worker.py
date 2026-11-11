# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Print the values of a patch of a written ``T`` as JSON, read through pybFoam.

``python _written_T_worker.py <system> <case> <time> <patch> <staging dir>``:
``<system>``, the case's ``constant/`` and ``<time>/T`` are staged as a case at
time ``0``, since pybFoam constructs no ``instant`` to set the time. A
subprocess: one ``Foam::Time`` per process.
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pybFoam as pyf
from pybFoam import volScalarField

RESULT_PREFIX = "#RESULT "


def main(system: Path, case: Path, time: str, patch: str, staged: Path) -> list[float]:
    shutil.copytree(system, staged / "system")
    shutil.copytree(case / "constant", staged / "constant")
    (staged / "0").mkdir()
    shutil.copy(case / time / "T", staged / "0" / "T")
    # Foam::Time and fvMesh keep raw references: the argList, Time and mesh stay bound
    arg_list = pyf.argList(["_written_T_worker", "-case", str(staged)])
    runtime = pyf.Time(arg_list)
    mesh = pyf.fvMesh(runtime)
    T = volScalarField.read_field(mesh, "T")
    return list(map(float, np.asarray(T[patch])))


if __name__ == "__main__":
    system, case, time, patch, staged = sys.argv[1:6]
    written = main(Path(system), Path(case), time, patch, Path(staged))
    print(RESULT_PREFIX + json.dumps({"T": written}), flush=True)
