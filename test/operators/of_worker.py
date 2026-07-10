# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Persistent pybFoam evaluation worker for one staged case.

Started as ``python of_worker.py <case_dir>`` by ``backends.Worker``; owns the
one ``Foam::Time`` of its process, reads the staged fields once, then serves
JSON-line requests from stdin (``{"func", "args", "scheme", "out"}``),
saving each result as ``<out>.npy`` and replying ``#RESULT {...}`` on stdout.

Operator notes:
- ``fvc.div(phi, T)``: pybFoam has no scalar convection overload, so the
  Gauss identity ``fvc.div(fvc.flux(phi, T, key="divT_<scheme>"))`` is used;
  the named keys are staged in ``system/fvSchemes`` (``fvc.flux`` only takes
  a lookup key, not a scheme string).
- ``fvm.*`` return the assembled matrix applied to the current field,
  ``M & psi``. ``operator&`` is not bound, but per component
  ``M & psi == A()*psi - H()`` (the component-average boundary-diagonal terms
  cancel). Like OpenFOAM's ``H()``, this zeroes invalid (empty-direction)
  components on 2D meshes.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pybFoam as pyf
from pybFoam import (
    fvc,
    fvm,
    fvScalarMatrix,
    fvVectorMatrix,
    volScalarField,
    volVectorField,
)

from schemes import div_scheme


def _internal(result: Any) -> np.ndarray:
    return np.asarray(result.ref().internalField())


def _matrix_apply(matrix: Any, psi: Any) -> np.ndarray:
    """M & psi from the bound accessors: A()*psi - H() (per unit volume)."""
    a = _internal(matrix.A())
    h = _internal(matrix.H())
    x = np.asarray(psi.internalField())
    return a[:, None] * x - h if x.ndim == 2 else a * x - h


def run(case_dir: Path) -> None:
    """Read the case once, then serve requests.

    Everything lives in this frame for the whole serve loop — the operator
    closures capture the fields, but ``arg_list``/``runtime``/``mesh`` must
    stay referenced too (pybFoam holds raw references across these objects).
    """
    os.chdir(case_dir)
    arg_list = pyf.argList(["ofOpsWorker"])
    runtime = pyf.Time(arg_list)
    mesh = pyf.fvMesh(runtime)

    t = volScalarField.read_field(mesh, "T")
    u = volVectorField.read_field(mesh, "U")
    gamma = volScalarField.read_field(mesh, "Gamma")
    phi = pyf.createPhi(u)

    ops: dict[tuple[str, tuple[str, ...]], Callable[[Any], np.ndarray]] = {
        ("fvc.interpolate", ("T",)): lambda s: _internal(fvc.interpolate(t)),
        ("fvc.flux", ("U",)): lambda s: _internal(fvc.flux(u)),
        ("fvc.grad", ("T",)): lambda s: _internal(fvc.grad(t)),
        ("fvc.div", ("phi",)): lambda s: _internal(fvc.div(phi)),
        ("fvc.div", ("phi", "T")): lambda s: _internal(
            fvc.div(fvc.flux(phi, t, key=f"divT_{s}"))
        ),
        ("fvc.div", ("phi", "U")): lambda s: _internal(
            fvc.div(phi, u, scheme=div_scheme(s, "U"))
        ),
        ("fvc.laplacian", ("Gamma", "T")): lambda s: _internal(fvc.laplacian(gamma, t)),
        ("fvc.laplacian", ("Gamma", "U")): lambda s: _internal(fvc.laplacian(gamma, u)),
        ("fvm.div", ("phi", "T")): lambda s: _matrix_apply(
            fvScalarMatrix(fvm.div(phi, t, scheme=div_scheme(s, "T"))), t
        ),
        ("fvm.div", ("phi", "U")): lambda s: _matrix_apply(
            fvVectorMatrix(fvm.div(phi, u, scheme=div_scheme(s, "U"))), u
        ),
        ("fvm.laplacian", ("Gamma", "T")): lambda s: _matrix_apply(
            fvScalarMatrix(fvm.laplacian(gamma, t)), t
        ),
        ("fvm.laplacian", ("Gamma", "U")): lambda s: _matrix_apply(
            fvVectorMatrix(fvm.laplacian(gamma, u)), u
        ),
    }
    _serve(ops)


def _serve(ops: dict[tuple[str, tuple[str, ...]], Callable[[Any], np.ndarray]]) -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request = json.loads(line)
        if request.get("exit"):
            return
        try:
            result = ops[(request["func"], tuple(request["args"]))](
                request.get("scheme")
            )
            np.save(request["out"], result)
            reply: dict[str, str] = {"status": "ok"}
        except Exception as exc:  # noqa: BLE001 — report to the client, keep serving
            reply = {"status": "error", "message": f"{type(exc).__name__}: {exc}"}
        print("#RESULT " + json.dumps(reply), flush=True)


def main() -> None:
    run(Path(sys.argv[1]).resolve())


if __name__ == "__main__":
    main()
