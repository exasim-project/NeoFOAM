# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Persistent pybFoam worker of the VoF parity tests, for one staged case.

Started as ``python _of_parity_worker.py <case_dir>``; owns the one
``Foam::Time`` of its process. At startup it generates the mesh from
``system/blockMeshDict``, runs setFields and setExprFields on it, prints
``#READY`` and serves JSON-line requests from stdin (``{"func", "args", "out",
"data"?}``), saving each array result as ``<out>.npy`` and replying
``#RESULT {...}`` on stdout.

``field.set`` copies a pushed array into the field, corrects the boundary
conditions and persists ``0/<name>`` (``writePrecision 17``, a bit-identical
round-trip) so the neon workers reload the same values. ``flux.update``
re-derives ``phi = fvc::flux(U)`` from the current ``U``.

Operator notes:
- ``vof.alphaPhaseFlux(alpha)`` (data ``[cAlpha]``): interFoam's ``alphaPhiUn``
  composed as ``models/mules.py`` does (vanLeer flux + compression flux, phic
  zeroed on the walls). There is no mixture, so ``nHatf`` follows
  ``interfaceProperties::calculateK`` from ``fvc.grad``/``fvc.interpolate``,
  with the per-face normalisation and ``& Sf`` done in numpy.
- ``vof.compressionVelocity(phi)`` (data ``[cAlpha]``) and
  ``vof.linearCompressionFlux(alpha)`` (with ``phir = phi``) are the two
  compression pieces of that composition, internal then boundary face values.
- ``mules.explicitSolve(alpha)`` (data ``[deltaT, cAlpha, nSteps, nCorr]``):
  nSteps of the explicit alphaEqn with nCorr correctors (flux from the current
  alpha, then ``mules.explicit_solve``). MULES reads psi0 from ``oldTime()``,
  which never advances here (the time index is fixed), so it is set to the
  field at the start of every step, as interFoam's time increment does.
  ``mules.correct(alpha, base)`` (data ``[deltaT, cAlpha]``): limits
  ``alphaPhiUn(alpha) - upwindFlux(base)`` with ``mules.correct``. Both return
  the new internal field and restore the previous one, so the field state the
  other tests see is unchanged.
- ``vof.courantNumbers(alpha)`` (data ``[deltaT]``): ``[CoNum, alphaCoNum]``
  from ``fvc.surfaceSum(mag(phi))`` as CourantNo.H and advectorVoF's
  alphaCourantNo.H compute them (interface cells: ``0.01 < alpha < 0.99``).

The fvMesh returned by ``generate_blockmesh`` is registered as ``region0`` and
must be dropped before constructing the disk-read ``fvMesh`` the operators use;
keeping both breaks field registration.
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pybFoam as pyf
from pybFoam import fvc, surfaceScalarField, volScalarField, volVectorField

Op = Callable[["np.ndarray | None"], "np.ndarray | None"]

_ALPHA_FIELDS = ("alpha", "alphaFixed", "alphaFixedShifted", "alphaSmooth")


@dataclass(frozen=True)
class _Case:
    """The time, mesh and face flux the operators share."""

    runtime: Any
    mesh: Any
    phi: Any


def _internal(result: Any) -> np.ndarray:
    return np.asarray(result.ref().internalField())


def _generate_mesh(case_dir: Path, runtime: Any) -> None:
    """Generate constant/polyMesh from system/blockMeshDict, in process."""
    block_dict = pyf.dictionary.read(str(case_dir / "system" / "blockMeshDict"))
    generated = pyf.meshing.generate_blockmesh(runtime, block_dict)
    del generated  # drop the registered region0 mesh before reading it back
    gc.collect()


def _write_initial_fields(case_dir: Path) -> None:
    """Run the case's field utilities on the generated mesh."""
    for tool in ("setFields", "setExprFields"):
        with (case_dir / f"log.{tool}").open("w") as log:
            subprocess.run([tool], cwd=case_dir, stdout=log, stderr=subprocess.STDOUT, check=True)


def _set_field(field: Any, values: np.ndarray | None) -> None:
    """Copy pushed values into the field and persist 0/<name> for neon."""
    assert values is not None
    view = np.asarray(field.internalField())
    view[:] = values.reshape(view.shape)
    field.correctBoundaryConditions()
    pyf.write(field)


def _nhatf(mesh: Any, phi: Any, alpha: Any) -> Any:
    """interfaceProperties::calculateK's nHatf = gradAlphaf/(|gradAlphaf| + deltaN) & Sf."""
    grad_alpha_f = _internal(
        fvc.interpolate(fvc.grad(alpha, scheme="Gauss linear"), scheme="linear")
    )
    delta_n = 1e-8 / np.cbrt(np.asarray(mesh.V()).mean())
    mag = np.sqrt(np.sum(grad_alpha_f * grad_alpha_f, axis=1))
    n_hat_fv = grad_alpha_f / (mag + delta_n)[:, None]
    # an oriented area field (like nHatfv & Sf) to hold the values; phic zeroes its boundary
    nhatf = surfaceScalarField(
        pyf.Word("nHatf"), phi / pyf.dimensionedScalar("unitVelocity", pyf.dimVelocity, 1.0)
    )
    np.asarray(nhatf.internalField())[:] = np.sum(
        n_hat_fv * np.asarray(mesh.Sf().internalField()), axis=1
    )
    return nhatf


def _with_boundary(field: Any) -> np.ndarray:
    """Internal then ``walls`` face values (the case's only patch, in NeoN's face order)."""
    return np.concatenate([np.asarray(field.internalField()), np.asarray(field["walls"])])


def _compression_velocity(mesh: Any, phi: Any, c_alpha: float) -> Any:
    """alphaEqn.H's phic = cAlpha*|phi/magSf|, zeroed on the non-coupled walls."""
    phic = surfaceScalarField(pyf.Word("phic"), pyf.mag(phi / mesh.magSf()) * c_alpha)
    phic["walls"] = pyf.scalarField([0.0] * len(phic["walls"]))
    return phic


def _linear_compression_flux(phir: Any, alpha: Any) -> Any:
    """fvc::flux(-fvc::flux(-phir, alpha2, linear), alpha, linear), alpha2 = 1 - alpha."""
    neg_phir = surfaceScalarField(pyf.Word("negPhir"), -phir)
    alpha2 = volScalarField(pyf.Word("alpha2"), -alpha + 1.0)
    phir_flux = surfaceScalarField(
        pyf.Word("phirFlux"), fvc.flux(neg_phir, alpha2, key="div(phirb,alpha)")
    )
    neg_phir_flux = surfaceScalarField(pyf.Word("negPhirFlux"), -phir_flux)
    return surfaceScalarField(
        pyf.Word("compressionFlux"), fvc.flux(neg_phir_flux, alpha, key="div(phirb,alpha)")
    )


def _alpha_phase_flux(mesh: Any, phi: Any, alpha: Any, c_alpha: float) -> Any:
    """alphaPhiUn as models/mules.py composes it (non-coupled walls: phic = 0)."""
    phic = _compression_velocity(mesh, phi, c_alpha)
    phir = surfaceScalarField(pyf.Word("phir"), phic * _nhatf(mesh, phi, alpha))
    return surfaceScalarField(
        pyf.Word("alphaPhiUn"),
        fvc.flux(phi, alpha, key="div(phi,alpha)") + _linear_compression_flux(phir, alpha),
    )


def _restoring(alpha: Any, step: Callable[[], None]) -> np.ndarray:
    """Run ``step`` on ``alpha``, return the result and restore the previous values."""
    view = np.asarray(alpha.internalField())
    previous = view.copy()
    step()
    result = view.copy()
    view[:] = previous
    alpha.correctBoundaryConditions()
    return result


def _mules_explicit(case: _Case, alpha: Any, data: Any) -> np.ndarray:
    delta_t, c_alpha, n_steps, n_corr = data
    case.runtime.setDeltaT(float(delta_t), False)

    def steps() -> None:
        for _ in range(int(n_steps)):
            alpha.oldTime().assign(alpha)
            for _ in range(int(n_corr)):
                alpha_phi = _alpha_phase_flux(case.mesh, case.phi, alpha, float(c_alpha))
                pyf.mules.explicit_solve(alpha, case.phi, alpha_phi)

    return _restoring(alpha, steps)


def _mules_correct(case: _Case, alpha: Any, base: Any, data: Any) -> np.ndarray:
    delta_t, c_alpha = data
    case.runtime.setDeltaT(float(delta_t), False)

    def step() -> None:
        alpha_phi_un = _alpha_phase_flux(case.mesh, case.phi, alpha, float(c_alpha))
        base_flux = fvc.flux(case.phi, base, key="divT_upwind")
        corr = surfaceScalarField(pyf.Word("alphaPhiCorr"), alpha_phi_un - base_flux)
        pyf.mules.correct(alpha, alpha_phi_un, corr)

    return _restoring(alpha, step)


def _courant_numbers(mesh: Any, phi: Any, alpha: Any, delta_t: float) -> np.ndarray:
    """CourantNo.H and advectorVoF's alphaCourantNo.H: 0.5*gMax(...)*deltaT."""
    sum_phi = np.array(
        volScalarField(pyf.Word("sumPhi"), fvc.surfaceSum(pyf.mag(phi))).internalField()
    )
    volumes = np.array(mesh.V())
    alpha_values = np.array(alpha.internalField())
    near_interface = (alpha_values > 0.01) & (alpha_values < 0.99)
    return np.array(
        [
            0.5 * np.max(sum_phi / volumes) * delta_t,
            0.5 * np.max(near_interface * sum_phi / volumes) * delta_t,
        ]
    )


def _alpha_ops(
    case: _Case, name: str, alpha: Any, alphas: dict[str, Any]
) -> dict[tuple[str, tuple[str, ...]], Op]:
    """Field access and the VoF/MULES ops of one phase-fraction field."""
    mesh, phi = case.mesh, case.phi
    ops: dict[tuple[str, tuple[str, ...]], Op] = {
        ("field.get", (name,)): lambda d: np.array(alpha.internalField()),
        ("field.set", (name,)): lambda d: _set_field(alpha, d),
        ("vof.alphaPhaseFlux", (name,)): lambda d: np.array(
            _alpha_phase_flux(mesh, phi, alpha, float(d[0])).internalField()
        ),
        ("vof.linearCompressionFlux", (name,)): lambda d: _with_boundary(
            _linear_compression_flux(phi, alpha)
        ),
        ("mules.explicitSolve", (name,)): lambda d: _mules_explicit(case, alpha, d),
        ("vof.courantNumbers", (name,)): lambda d: _courant_numbers(mesh, phi, alpha, float(d[0])),
    }
    for base_name, base in alphas.items():
        ops[("mules.correct", (name, base_name))] = lambda d, base=base: _mules_correct(
            case, alpha, base, d
        )
    return ops


def run(case_dir: Path) -> None:
    """Generate the mesh and inputs, read the fields, then serve requests.

    Everything lives in this frame for the whole serve loop: the operator
    closures capture the fields, but ``arg_list``/``runtime``/``mesh`` must
    stay referenced too (pybFoam holds raw references across these objects).
    """
    os.chdir(case_dir)
    arg_list = pyf.argList(["ofVofParityWorker"])
    runtime = pyf.Time(arg_list)
    _generate_mesh(case_dir, runtime)
    _write_initial_fields(case_dir)
    mesh = pyf.fvMesh(runtime)

    u = volVectorField.read_field(mesh, "U")
    velocities = {
        name: volVectorField.read_field(mesh, name) for name in ("URotation", "UTranslation")
    }
    alphas = {name: volScalarField.read_field(mesh, name) for name in _ALPHA_FIELDS}
    phi = pyf.createPhi(u)
    case = _Case(runtime, mesh, phi)
    centres = np.array(mesh.C().internalField())
    volumes = np.array(mesh.V())

    ops: dict[tuple[str, tuple[str, ...]], Op] = {
        ("mesh.C", ()): lambda d: centres,
        ("mesh.V", ()): lambda d: volumes,
        ("field.get", ("U",)): lambda d: np.array(u.internalField()),
        ("field.set", ("U",)): lambda d: _set_field(u, d),
        ("flux.update", ("U",)): lambda d: phi.assign(fvc.flux(u)),
        ("vof.compressionVelocity", ("phi",)): lambda d: _with_boundary(
            _compression_velocity(mesh, phi, float(d[0]))
        ),
    }
    for name, velocity in velocities.items():
        ops[("field.get", (name,))] = lambda d, velocity=velocity: np.array(
            velocity.internalField()
        )
    for name, alpha in alphas.items():
        ops |= _alpha_ops(case, name, alpha, alphas)
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
            result = ops[(request["func"], tuple(request["args"]))](data)
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
