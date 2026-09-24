# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Evaluate ``compute_alpha_courant_number`` on one time directory of a case
that native ``interFoam`` has already run.

Run as ``python _native_alpha_courant_worker.py <case_dir> <time_name>
<delta_t>``; writes ``<case_dir>/alphaCo.<time_name>.json`` holding
``[alphaCoNum, meanAlphaCo]``.

One ``Foam::Time`` per process is a hard OpenFOAM constraint, and the time a
``Foam::Time`` opens at is a ``system/controlDict`` entry consumed by its
constructor — so one process evaluates exactly one time and the caller spawns
one worker per time. ``startTime`` is set through pybFoam's own ``dictionary``
reader/writer (the case file is never text-patched); ``alpha.water`` and ``phi``
are then read back from that directory exactly as interFoam wrote them
(``writeFormat binary``, so bit-identical), and the run's fixed ``deltaT`` is
re-imposed before the call — reproducing the state ``alphaCourantNo.H`` saw at
the top of the following time step.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pybFoam as pyf
from pybFoam import surfaceScalarField, volScalarField

from neofoam.solver.incompressibleVoF.incompressibleVoF import (
    compute_alpha_courant_number,
)


def _open_at(case_dir: Path, time_name: str) -> pyf.Time:
    """Point ``system/controlDict`` at ``time_name`` and construct the Time."""
    control_dict_path = case_dir / "system" / "controlDict"
    control_dict = pyf.dictionary.read(str(control_dict_path))
    control_dict.set("startFrom", "startTime")
    control_dict.set("startTime", float(time_name))
    control_dict.write(str(control_dict_path))

    runtime = pyf.Time(pyf.argList(["nativeAlphaCourantWorker"]))
    assert runtime.timeName() == time_name, f"opened {runtime.timeName()!r}, expected {time_name!r}"
    return runtime


def run(case_dir: Path, time_name: str, delta_t: float) -> list[float]:
    """Return ``[alphaCoNum, meanAlphaCo]`` for the state written at ``time_name``."""
    runtime = _open_at(case_dir, time_name)
    mesh = pyf.fvMesh(runtime)

    alpha1 = volScalarField.read_field(mesh, "alpha.water")
    phi = surfaceScalarField.read_field(mesh, "phi")
    runtime.setDeltaT(delta_t)

    return list(compute_alpha_courant_number(phi, alpha1))


def main() -> None:
    case_dir = Path(sys.argv[1]).resolve()
    time_name = sys.argv[2]
    delta_t = float(sys.argv[3])
    os.chdir(case_dir)
    (case_dir / f"alphaCo.{time_name}.json").write_text(
        json.dumps(run(case_dir, time_name, delta_t))
    )


if __name__ == "__main__":
    main()
