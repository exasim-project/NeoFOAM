# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Batch ``set_time_step`` evaluator for one staged case (one ``Foam::Time``).

Run as ``python _set_time_step_worker.py <case_dir> <request.json>``; writes
``<case_dir>/result.json``. One process owns exactly one ``Foam::Time`` and one
mesh (the ``row4`` mesh from ``cases/alphaCourant``, reused as-is): U/alpha1 are
re-seeded per scenario (uniform fields, see ``test_set_time_step.py`` for why
that makes both Courant numbers exact linear functions of ``u_x``/``dt0``),
``system/controlDict`` is rewritten from a pristine template via pybFoam's own
``dictionary``/``set``/``write`` (never text-patched), and ``set_time_step`` is
called directly on the live fields/runtime.

A ``_SpyRuntime`` wraps the real ``Foam::Time`` passed as the ``runtime``
argument so ``setDeltaT`` calls can be counted (proves the early-return path
never mutates dt) without touching production code.

The request is ``{"scenarios": {name: {"u_x": float, "alpha": float,
"dt0": float, "advance": bool, "control_dict": {key: value, ...}}}}`` —
``advance`` increments the ``Foam::Time`` once first (so ``timeIndex() != 0``
and ``setInitialDeltaT.H`` is gated off), and ``control_dict`` only
carries the keys that should be *present* in ``system/controlDict``; a key
that must be *absent* (to exercise a default) is simply left out.
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

from neofoam.framework.context import Context
from neofoam.solver.incompressibleVoF.incompressibleVoF import set_time_step


def _generate_mesh(case_dir: Path, runtime: Any) -> None:
    """Generate constant/polyMesh from system/blockMeshDict, in process."""
    block_dict = pyf.dictionary.read(str(case_dir / "system" / "blockMeshDict"))
    generated = pyf.meshing.generate_blockmesh(runtime, block_dict)
    del generated  # drop the registered region0 mesh before reading it back
    gc.collect()


class _SpyRuntime:
    """Stands in for the ``runtime`` argument: counts ``setDeltaT`` calls while
    forwarding everything to the real ``Foam::Time``."""

    def __init__(self, real: Any) -> None:
        self._real = real
        self.set_delta_t_calls: list[float] = []

    def deltaTValue(self) -> float:
        return float(self._real.deltaTValue())

    def timeIndex(self) -> int:
        return int(self._real.timeIndex())

    def setDeltaT(self, value: float) -> None:
        self.set_delta_t_calls.append(float(value))
        self._real.setDeltaT(value)


def run(case_dir: Path, request: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Evaluate every scenario on the row4 mesh; keep the frame alive."""
    arg_list = pyf.argList(["setTimeStepWorker"])
    runtime = pyf.Time(arg_list)
    _generate_mesh(case_dir, runtime)
    mesh = pyf.fvMesh(runtime)

    u = volVectorField.read_field(mesh, "U")
    phi = pyf.createPhi(u)
    alpha1 = volScalarField.read_field(mesh, "alpha.water")

    template = pyf.dictionary.read(str(case_dir / "controlDict.template"))
    control_dict_path = case_dir / "system" / "controlDict"
    ctx = Context(fields={}, models={})

    results: dict[str, dict[str, Any]] = {}
    for name, spec in request["scenarios"].items():
        u_view = np.asarray(u.internalField())
        u_view[:] = [spec["u_x"], 0.0, 0.0]
        u.correctBoundaryConditions()
        phi.assign(fvc.flux(u))

        alpha_view = np.asarray(alpha1.internalField())
        alpha_view[:] = spec["alpha"]
        alpha1.correctBoundaryConditions()

        # Fresh copy of the pristine template every scenario: only the keys
        # named in spec["control_dict"] are set, so an omitted key reproduces
        # the "absent from controlDict" case exactly.
        control_dict = pyf.dictionary(template)
        for key, value in spec["control_dict"].items():
            control_dict.set(key, value)
        control_dict.write(str(control_dict_path))

        # "advance": one increment first, so timeIndex() != 0 and the scenario
        # exercises a mid-run step (setInitialDeltaT.H gated off) rather than the
        # first one. Such scenarios must come last in the request.
        if spec.get("advance"):
            runtime.increment()

        runtime.setDeltaT(spec["dt0"])
        spy = _SpyRuntime(runtime)
        set_time_step(ctx, phi, alpha1, spy)

        results[name] = {
            "final_dt": runtime.deltaTValue(),
            "set_delta_t_calls": spy.set_delta_t_calls,
        }
    return results


def main() -> None:
    case_dir = Path(sys.argv[1]).resolve()
    request = json.loads(Path(sys.argv[2]).read_text())
    os.chdir(case_dir)
    (case_dir / "result.json").write_text(json.dumps(run(case_dir, request)))


if __name__ == "__main__":
    main()
