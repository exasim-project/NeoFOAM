# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Subprocess worker for ``test_sampling_e2e``: evaluate surface pipelines on a real mesh.

One ``Foam::Time`` per process (and the test process already built the mesh), so
the pipelines are evaluated here and only their numbers travel back, as JSON.

The two probe fields are seeded analytically through the numpy view rather than
solved for, which is what makes the expected values in the test analytic: ``p``
is the cell-centre *x* coordinate, so its ``p == x0`` iso-surface is exactly the
plane ``x = x0``, and ``U = (x, y, 0)``, so a ``cell``-scheme sample of ``|U|``
on a plane is the cell value of every cut cell.

Run it as ``python _sampling_driver.py <case> <out.json>``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pybFoam as pyf

from neofoam.framework.context import Context
from neofoam.postprocess import Area, Mag, Mean, Pipeline, Sample, Sum, iso_surface, plane

#: Cut through the middle cell column of the cavity, normal to x.
PLANE_POINT = (0.05, 0.0, 0.0)
PLANE_NORMAL = (1.0, 0.0, 0.0)

#: The iso-value of the seeded ``p == x`` field: the same plane, cut differently.
ISO_VALUE = 0.05


def pipelines() -> dict[str, Pipeline]:
    """The named tables the test asserts on."""
    return {
        "cut_area": plane(PLANE_POINT, PLANE_NORMAL) | Sum(name="cut_area"),
        "cut_area_via_area": plane(PLANE_POINT, PLANE_NORMAL) | Area() | Sum(name="cut_area"),
        "mean_face_area": plane(PLANE_POINT, PLANE_NORMAL) | Mean(name="mean_face_area"),
        "iso_area": iso_surface("p", ISO_VALUE) | Sum(name="iso_area"),
        "iso_p": iso_surface("p", ISO_VALUE, field="p") | Mean(name="iso_p"),
        "mean_speed_from_source": (
            plane(PLANE_POINT, PLANE_NORMAL, field="U", scheme="cell")
            | Mag()
            | Mean(name="mean_speed")
        ),
        "mean_speed_from_sample": (
            plane(PLANE_POINT, PLANE_NORMAL, scheme="cell")
            | Sample(field="U")
            | Mag()
            | Mean(name="mean_speed")
        ),
    }


def open_case(case: Path) -> "pyf.Time":
    """The case's ``Foam::Time`` — the caller must keep it alive (see :func:`probe_context`)."""
    # A directly-constructed Time skips argList, which is what exports these.
    os.environ["FOAM_CASE"] = str(case)
    os.environ["FOAM_CASENAME"] = case.name
    return pyf.Time(str(case.parent), case.name)


def probe_context(runtime: "pyf.Time") -> Context:
    """A Context on *runtime* holding ``p = x`` and ``U = (x, y, 0)`` (see the module docstring).

    *runtime* stays with the caller: the mesh's object registry is a child of the
    ``Time``, and an iso-surface looks its field up there, so letting the ``Time``
    fall out of scope aborts the run.
    """
    mesh = pyf.fvMesh(runtime)

    centres = np.asarray(mesh.C().internalField())
    p = pyf.volScalarField.read_field(mesh, "p")
    np.asarray(p.internalField())[:] = centres[:, 0]
    p.correctBoundaryConditions()
    velocity = pyf.volVectorField.read_field(mesh, "U")
    np.asarray(velocity.internalField())[:] = np.column_stack(
        [centres[:, 0], centres[:, 1], np.zeros(len(centres))]
    )
    velocity.correctBoundaryConditions()

    return Context(fields={"p": p, "U": velocity}, models={}, mesh=mesh)


def main() -> None:
    case, out = Path(sys.argv[1]), Path(sys.argv[2])
    runtime = open_case(case)
    ctx = probe_context(runtime)
    results = {
        name: pipeline.compute(ctx).values[0].value for name, pipeline in pipelines().items()
    }
    out.write_text(json.dumps(results))


if __name__ == "__main__":
    main()
