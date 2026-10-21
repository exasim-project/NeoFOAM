# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Record what one isoAdvector ``alpha_advection`` call does to ``Time`` and ``U``.

Run as ``python _iso_advector_worker.py <case_dir>``; writes
``<case_dir>/iso_advector.json``. A process owns exactly one ``Foam::Time``,
hence a worker: the dump is about what happens *inside* one call.

The advector is swapped for a recorder that logs the ``Foam::Time`` state and
the velocity field at the moment ``advect()`` is entered, then delegates to the
real ``Foam::isoAdvection`` — that instant is the one isoAdvection interpolates
``U`` at to get the interface normal velocity, so it is exactly what native's
``U -= fvc::reconstruct(mesh.phi())`` bracket in ``alphaEqn.H`` is there to set.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pybFoam as pyf
from pybFoam import fvc, volVectorField

from neofoam.solver.incompressibleVoF.create_fields import create_init
from neofoam.solver.incompressibleVoF.models.alpha_advection.models import iso_advector


class _RecordingAdvector:
    """Logs the Time state and U each ``advect()`` sees, then calls the real one."""

    def __init__(self, advector: Any, runtime: Any, U: Any) -> None:
        self._advector = advector
        self._runtime = runtime
        self._U = U
        self.seen: list[dict[str, Any]] = []

    def advect(self) -> None:
        self.seen.append(
            {
                "time": self._runtime.value(),
                "deltaT": self._runtime.deltaTValue(),
                "timeIndex": self._runtime.timeIndex(),
                "U": np.asarray(self._U.internalField()).tolist(),
            }
        )
        self._advector.advect()

    def get_rho_phi(self, rho1: Any, rho2: Any) -> Any:
        return self._advector.get_rho_phi(rho1, rho2)


def run(case_dir: Path) -> dict[str, Any]:
    runner = create_init(case_dir=case_dir)
    runner.argv = ["incompressibleVoF"]
    ctx = runner.run()

    runtime = ctx.models["runtime"]
    runtime.increment()  # one real time step, as the solver's time loop does

    mesh = ctx.mesh
    if mesh.dynamic():
        # pimpleAlgorithm's mesh_update step: the alpha equation always runs on
        # an already-moved mesh, and mesh.phi() only exists after the move.
        mesh.updateMesh()

    U = ctx.fields["U"]
    recorder = _RecordingAdvector(ctx.models["advector"], runtime, U)

    before = {
        "time": runtime.value(),
        "deltaT": runtime.deltaTValue(),
        "timeIndex": runtime.timeIndex(),
    }
    mesh_velocity = (
        np.asarray(
            volVectorField(pyf.Word("meshUCheck"), fvc.reconstruct(mesh.phi())).internalField()
        ).tolist()
        if mesh.moving()
        else None
    )
    U_before = np.asarray(U.internalField()).tolist()

    iso_advector.alpha_advection(
        ctx.fields["alpha1"],
        ctx.fields["alpha2"],
        ctx.fields["phi"],
        ctx.fields["rhoPhi"],
        ctx.fields["rho"],
        U,
        ctx.models["mixture"],
        recorder,
    )

    return {
        "moving": mesh.moving(),
        "before": before,
        "sub_steps": recorder.seen,
        "after": {
            "time": runtime.value(),
            "deltaT": runtime.deltaTValue(),
            "timeIndex": runtime.timeIndex(),
        },
        "mesh_velocity": mesh_velocity,
        "U_before": U_before,
        "U_after": np.asarray(U.internalField()).tolist(),
    }


if __name__ == "__main__":
    case = Path(sys.argv[1]).resolve()
    os.chdir(case)
    (case / "iso_advector.json").write_text(json.dumps(run(case), indent=1))
