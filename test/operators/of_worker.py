# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Persistent pybFoam evaluation worker for one staged case.

Started as ``python of_worker.py <case_dir>`` by ``backends.Worker``; owns the
one ``Foam::Time`` of its process. At startup it generates the mesh from
``system/blockMeshDict`` via ``pybFoam.meshing.generate_blockmesh`` (no
external ``blockMesh`` binary), prints ``#READY`` and serves JSON-line
requests from stdin (``{"func", "args", "scheme", "out", "data"?}``), saving
each array result as ``<out>.npy`` and replying ``#RESULT {...}`` on stdout.

Field values come from the tests: ``field.set`` copies a pushed numpy array
into the field's ``internalField``, corrects the boundary conditions, and
persists ``0/<name>`` (``writePrecision 17`` — bit-identical round-trip) so
the neon workers can reload exactly the same values. ``flux.update``
re-derives ``phi = fvc::flux(U)`` from the current ``U``.

Operator notes:
- ``fvc.div(phi, T)``: pybFoam has no scalar convection overload, so the
  Gauss identity ``fvc.div(fvc.flux(phi, T, key="divT_<scheme>"))`` is used;
  the named keys are staged in ``system/fvSchemes`` (``fvc.flux`` only takes
  a lookup key, not a scheme string).
- ``fvm.*`` return the assembled matrix applied to the current field,
  ``M & psi``. ``operator&`` is not bound, but per component
  ``M & psi == A()*psi - H()`` (the component-average boundary-diagonal terms
  cancel).

Staging notes:
- the fvMesh returned by ``generate_blockmesh`` is registered as ``region0``
  and must be dropped before constructing the disk-read ``fvMesh`` the
  operators use — keeping both breaks field registration.
"""

from __future__ import annotations

import gc
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

Op = Callable[[str, "np.ndarray | None"], "np.ndarray | None"]


def _internal(result: Any) -> np.ndarray:
    return np.asarray(result.ref().internalField())


def _matrix_apply(matrix: Any, psi: Any) -> np.ndarray:
    """M & psi from the bound accessors: A()*psi - H() (per unit volume)."""
    a = _internal(matrix.A())
    h = _internal(matrix.H())
    x = np.asarray(psi.internalField())
    return a[:, None] * x - h if x.ndim == 2 else a * x - h


def _generate_mesh(case_dir: Path, runtime: Any) -> None:
    """Generate constant/polyMesh from system/blockMeshDict, in process."""
    block_dict = pyf.dictionary.read(str(case_dir / "system" / "blockMeshDict"))
    generated = pyf.meshing.generate_blockmesh(runtime, block_dict)
    del generated  # drop the registered region0 mesh before reading it back
    gc.collect()


def _set_field(field: Any, values: np.ndarray | None) -> None:
    """Copy pushed values into the field and persist 0/<name> for neon."""
    assert values is not None
    view = np.asarray(field.internalField())
    view[:] = values.reshape(view.shape)
    field.correctBoundaryConditions()
    pyf.write(field)


def run(case_dir: Path) -> None:
    """Generate the mesh, read the fields, then serve requests.

    Everything lives in this frame for the whole serve loop — the operator
    closures capture the fields, but ``arg_list``/``runtime``/``mesh`` must
    stay referenced too (pybFoam holds raw references across these objects).
    """
    os.chdir(case_dir)
    arg_list = pyf.argList(["ofOpsWorker"])
    runtime = pyf.Time(arg_list)
    _generate_mesh(case_dir, runtime)
    mesh = pyf.fvMesh(runtime)

    t = volScalarField.read_field(mesh, "T")
    u = volVectorField.read_field(mesh, "U")
    gamma = volScalarField.read_field(mesh, "Gamma")
    phi = pyf.createPhi(u)
    centres = np.array(mesh.C().internalField())

    ops: dict[tuple[str, tuple[str, ...]], Op] = {
        ("mesh.C", ()): lambda s, d: centres,
        ("field.get", ("T",)): lambda s, d: np.array(t.internalField()),
        ("field.get", ("U",)): lambda s, d: np.array(u.internalField()),
        ("field.get", ("Gamma",)): lambda s, d: np.array(gamma.internalField()),
        ("field.set", ("T",)): lambda s, d: _set_field(t, d),
        ("field.set", ("U",)): lambda s, d: _set_field(u, d),
        ("field.set", ("Gamma",)): lambda s, d: _set_field(gamma, d),
        ("flux.update", ("U",)): lambda s, d: phi.assign(fvc.flux(u)),
        ("fvc.interpolate", ("T",)): lambda s, d: _internal(fvc.interpolate(t)),
        ("fvc.flux", ("U",)): lambda s, d: _internal(fvc.flux(u)),
        ("fvc.grad", ("T",)): lambda s, d: _internal(fvc.grad(t)),
        ("fvc.div", ("phi",)): lambda s, d: _internal(fvc.div(phi)),
        ("fvc.div", ("phi", "T")): lambda s, d: _internal(
            fvc.div(fvc.flux(phi, t, key=f"divT_{s}"))
        ),
        ("fvc.div", ("phi", "U")): lambda s, d: _internal(
            fvc.div(phi, u, scheme=div_scheme(s, "U"))
        ),
        ("fvc.laplacian", ("Gamma", "T")): lambda s, d: _internal(
            fvc.laplacian(gamma, t)
        ),
        ("fvc.laplacian", ("Gamma", "U")): lambda s, d: _internal(
            fvc.laplacian(gamma, u)
        ),
        ("fvm.div", ("phi", "T")): lambda s, d: _matrix_apply(
            fvScalarMatrix(fvm.div(phi, t, scheme=div_scheme(s, "T"))), t
        ),
        ("fvm.div", ("phi", "U")): lambda s, d: _matrix_apply(
            fvVectorMatrix(fvm.div(phi, u, scheme=div_scheme(s, "U"))), u
        ),
        ("fvm.laplacian", ("Gamma", "T")): lambda s, d: _matrix_apply(
            fvScalarMatrix(fvm.laplacian(gamma, t)), t
        ),
        ("fvm.laplacian", ("Gamma", "U")): lambda s, d: _matrix_apply(
            fvVectorMatrix(fvm.laplacian(gamma, u)), u
        ),
    }
    print("#READY", flush=True)
    _serve(ops)


def _serve(ops: dict[tuple[str, tuple[str, ...]], Op]) -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request = json.loads(line)
        if request.get("exit"):
            return
        try:
            data = np.load(request["data"]) if request.get("data") else None
            result = ops[(request["func"], tuple(request["args"]))](
                request.get("scheme"), data
            )
            if result is not None:
                np.save(request["out"], result)
            reply: dict[str, str] = {"status": "ok"}
        except Exception as exc:  # noqa: BLE001 — report to the client, keep serving
            reply = {"status": "error", "message": f"{type(exc).__name__}: {exc}"}
        print("#RESULT " + json.dumps(reply), flush=True)


def main() -> None:
    run(Path(sys.argv[1]).resolve())


if __name__ == "__main__":
    main()
