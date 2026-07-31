# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Run the incompressibleVoF staged init once and dump what it built.

Run as ``python _create_fields_worker.py <case_dir>``; writes
``<case_dir>/create_fields.json``. One process owns exactly one ``Foam::Time``,
so the whole LOAD -> RESOLVE -> BUILD pipeline is executed here in a single pass
and every quantity the tests assert on is serialized to JSON.

The pipeline is driven exactly as production does it (``SolverSpec._run_initialize``):
``create_init(case_dir)`` -> set ``runner.argv`` -> ``runner.run()``.

``runtime.write(True)`` at the end is the solver's own write path
(``write_output``): it makes OpenFOAM write the AUTO_WRITE fields the pipeline
registered into the ``0/`` directory, which is the only way boundary-condition
types and dimension sets of the constructed fields are observable from Python
(pybFoam binds neither ``boundaryField()`` nor ``dimensions()`` on a
GeometricField).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pybFoam as pyf

from neofoam.solver.incompressibleVoF.create_fields import create_init


def _internal(field: Any) -> Any:
    """Internal field values as plain Python lists (scalars or vectors).

    ``hRef`` is a ``uniformDimensionedScalarField`` — a single dimensioned
    constant, not a ``GeometricField`` — so it has no ``internalField()``;
    return its scalar value directly instead.
    """
    if isinstance(field, pyf.uniformDimensionedScalarField):
        return field.value()
    return np.asarray(field.internalField()).tolist()


def run(case_dir: Path) -> dict[str, Any]:
    runner = create_init(case_dir=case_dir)
    runner.argv = ["incompressibleVoF"]
    ctx = runner.run()

    mixture = ctx.models["mixture"]
    result: dict[str, Any] = {
        "field_keys": sorted(ctx.fields),
        "model_keys": sorted(ctx.models),
        "write_fields": sorted(ctx.write_fields),
        "n_cells": ctx.mesh.nCells(),
        "n_internal_faces": ctx.mesh.nInternalFaces(),
        "mesh_dynamic": ctx.mesh.dynamic(),
        "mesh_type": type(ctx.mesh).__name__,
        # None on a static mesh (createUfIfPresent.H builds Uf only when the
        # mesh can move); its name otherwise.
        "Uf": None if ctx.models["Uf"] is None else str(ctx.models["Uf"].name()),
        "dynamic_mesh_controls": ctx.models["dynamic_mesh_controls"],
        # ``uniformDimensionedScalarField.name()`` (hRef) returns a ``Word``,
        # not a plain ``str`` like ``GeometricField.name()`` does — ``str()``
        # normalises both for JSON.
        "registered_names": {key: str(ctx.fields[key].name()) for key in sorted(ctx.fields)},
        "internal": {key: _internal(ctx.fields[key]) for key in sorted(ctx.fields)},
        "mixture": {
            "rho1": mixture.rho1().value(),
            "rho2": mixture.rho2().value(),
            "alpha1_name": mixture.alpha1().name(),
            "alpha2_name": mixture.alpha2().name(),
        },
        "pressure_reference": ctx.models["pressure_reference"],
        "pimple_control": {
            "nOuterCorrectors": ctx.models["pimple_control"].nOuterCorrectors,
            "nCorrectors": ctx.models["pimple_control"].nCorrectors,
            "momentumPredictor": ctx.models["pimple_control"].momentumPredictor_enabled,
        },
        "turbulence_type": type(ctx.models["turbulence"]).__name__,
        "cumulativeContErr": ctx.models["cumulativeContErr"],
    }

    # Solver write path (``write_output``): hand the AUTO_WRITE fields to
    # OpenFOAM's own writer. The clock is advanced one step first so the fields
    # land in a *new* time directory — a file in ``0/`` could not be told apart
    # from the case input the pipeline read.
    runtime = ctx.models["runtime"]
    runtime.increment()
    runtime.write(True)
    result["written_time"] = str(runtime.timeName())
    return result


def main() -> None:
    case_dir = Path(sys.argv[1]).resolve()
    os.chdir(case_dir)
    (case_dir / "create_fields.json").write_text(json.dumps(run(case_dir), indent=1))


if __name__ == "__main__":
    main()
