# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Persistent pybFoam evaluation worker for one staged case.

Started as ``python of_worker.py <case_dir>`` by ``backends.Worker``; owns the
one ``Foam::Time`` of its process. At startup it stages the case itself:
generates the mesh from ``system/blockMeshDict`` via
``pybFoam.meshing.generate_blockmesh`` (no external ``blockMesh`` binary) and
seeds the deterministic analytic ``T``/``U`` fields (written with
``writePrecision 17``, so the values round-trip bit-identically — the neon
workers read exactly the same numbers from disk). Once staged it prints
``#READY`` and serves JSON-line requests from stdin
(``{"func", "args", "scheme", "out"}``), saving each result as ``<out>.npy``
and replying ``#RESULT {...}`` on stdout.

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
- the seed fields are smooth, non-symmetric functions of the cell centres so
  every operator produces a non-trivial result.
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


def _seed_fields(t: Any, u: Any, centres: np.ndarray) -> None:
    """Seed deterministic analytic values into T and U and write them to 0/."""
    span = centres.max(axis=0) - centres.min(axis=0)
    scale = float(span.max())
    x, y, z = (centres[:, i] / scale for i in range(3))

    t_view = np.asarray(t.internalField())
    t_view[:] = 2.0 + np.sin(np.pi * x) * np.cos(np.pi * y) + 0.3 * z

    # nonzero divergence in every direction (U_i must vary with x_i)
    u_view = np.asarray(u.internalField())
    u_view[:, 0] = 1.0 + np.sin(np.pi * y) + 0.5 * np.sin(np.pi * x)
    u_view[:, 1] = 0.5 + np.cos(np.pi * x) + 0.5 * np.cos(np.pi * y)
    u_view[:, 2] = 0.1 + 0.2 * np.sin(np.pi * z)

    t.correctBoundaryConditions()
    u.correctBoundaryConditions()
    pyf.write(t)
    pyf.write(u)


def run(case_dir: Path) -> None:
    """Stage the case, then serve requests.

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
    _seed_fields(t, u, np.asarray(mesh.C().internalField()))
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
    print("#READY", flush=True)
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
