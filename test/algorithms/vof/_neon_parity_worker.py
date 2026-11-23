# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Persistent NeoN worker of the VoF parity tests, for one staged case and executor.

Started as ``python _neon_parity_worker.py <case_dir> <executor>``; owns the
one ``Foam::Time`` and the Kokkos runtime of its process, reads the staged
fields once, then serves JSON-line requests from stdin (``{"func", "args",
"out", "data"?}``), saving each result as ``<out>.npy`` and replying
``#RESULT {...}`` on stdout.

After the pybFoam worker persists ``0/<name>``, a ``field.reload`` request
re-reads the field from disk into a fresh NeoN field on this worker's executor
(host to device copy on GPU). ``flux.update`` rebuilds ``phi`` from the on-disk
``U`` exactly like startup (``nfb.create_phi`` runs OpenFOAM's ``fvc::flux``),
so both backends consume a bit-identical flux. The live fields are looked up
per request from the ``fields`` dict, never captured in the op closures.

Operator notes:
- ``vof.*`` / ``mules.*`` run the NeoFOAM VoF bindings (``alpha_phase_flux``,
  ``interface_compression_velocity``, ``linear_compression_flux``,
  ``upwind_flux``, ``courant_numbers``, ``mules_explicit_solve``,
  ``mules_correct``) with the same data payloads as the pybFoam worker; the
  MULES ops return the new internal field and then reload the case's one from
  disk. With the ``into``/``composed`` flag the flux ops use the ``out=``
  variants, writing into a buffer of stale values; composed ``alphaPhaseFlux``
  is ``vanleer_flux``, ``interface_nhatf`` and ``add_interface_compression_flux``
  into one buffer.
- all results are copied to host; every NeoN object stays local to ``serve``
  so teardown happens before ``nn.finalize()``.
"""

from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

import neon._neon as nn
import numpy as np
import pybFoam as pyf

from neofoam import neofoam_bindings as nfb

Op = Callable[["np.ndarray | None"], "np.ndarray | None"]

_ALPHA_FIELDS = ("alpha", "alphaFixed", "alphaFixedShifted", "alphaSmooth")


def _to_numpy(vec: Any) -> np.ndarray:
    return np.asarray(vec.copy_to_host())


def _with_boundary(field: Any) -> np.ndarray:
    return np.concatenate(
        [_to_numpy(field.internal_vector()), _to_numpy(field.boundary_data_value())]
    )


def serve(case_dir: Path, executor: str) -> None:
    """Read the case once, then serve requests.

    Everything lives in this frame for the whole serve loop: ``arg_list`` and
    ``run_time`` must stay referenced (pybFoam holds raw references), and all
    NeoN objects must be destroyed when this frame ends, before ``main``
    finalizes Kokkos.
    """
    os.chdir(case_dir)
    arg_list = pyf.argList(["neonVofParityWorker"])
    run_time = pyf.Time(arg_list)
    rt = nfb.create_adapter_run_time(run_time, executor)

    fields: dict[str, Any] = {
        "U": nfb.read_vector_volume_field(rt, "U"),
        "phi": nfb.create_phi(rt, "U"),
    }
    for name in _ALPHA_FIELDS:
        fields[name] = nfb.read_scalar_volume_field(rt, name)

    def reload_scalar(name: str) -> None:
        fields[name] = nfb.read_scalar_volume_field(rt, name)

    def reload_velocity() -> None:
        fields["U"] = nfb.read_vector_volume_field(rt, "U")

    def update_flux() -> None:
        fields["phi"] = nfb.create_phi(rt, "U")

    def phase_flux(name: str, c_alpha: float) -> Any:
        return nfb.alpha_phase_flux(rt, fields[name], fields["phi"], c_alpha)

    def stale_buffer() -> Any:
        """A face field of stale non-zero values for an ``out=`` variant to overwrite."""
        return fields["phi"] * 7.0

    def compression_velocity(data: Any) -> np.ndarray:
        c_alpha, into = float(data[0]), bool(data[1])
        if not into:
            return _with_boundary(nfb.interface_compression_velocity(fields["phi"], c_alpha))
        out = stale_buffer()
        nfb.interface_compression_velocity(fields["phi"], c_alpha, out=out)
        return _with_boundary(out)

    def linear_compression_flux(name: str, data: Any) -> np.ndarray:
        if not bool(data[0]):
            return _with_boundary(nfb.linear_compression_flux(rt, fields[name], fields["phi"]))
        out = stale_buffer()
        nfb.linear_compression_flux(rt, fields[name], fields["phi"], out=out)
        return _with_boundary(out)

    def composed_phase_flux(name: str, c_alpha: float) -> Any:
        """alphaPhiUn from the out= bindings, as the NeoN MULES ops compose it."""
        alpha, phi = fields[name], fields["phi"]
        out, phic, nhatf = stale_buffer(), stale_buffer(), stale_buffer()
        nfb.vanleer_flux(rt, alpha, phi, out=out)
        nfb.interface_compression_velocity(phi, c_alpha, out=phic)
        nfb.interface_nhatf(rt, alpha, out=nhatf)
        nfb.add_interface_compression_flux(rt, alpha, phic, nhatf, out)
        return out

    def restoring(name: str, step: Callable[[Any], None]) -> np.ndarray:
        """Run ``step`` on the field, return the result, reload the case's field."""
        step(fields[name])
        result = _to_numpy(fields[name].internal_vector())
        fields[name] = nfb.read_scalar_volume_field(rt, name)
        return result

    def mules_explicit(name: str, data: Any) -> np.ndarray:
        delta_t, c_alpha, n_steps, n_corr = data
        alpha_old = nfb.read_scalar_volume_field(rt, name)

        def steps(alpha: Any) -> None:
            for _ in range(int(n_steps)):
                alpha_old.assign(alpha)
                for _ in range(int(n_corr)):
                    alpha_phi = phase_flux(name, float(c_alpha))
                    nfb.mules_explicit_solve(
                        alpha, alpha_old, fields["phi"], alpha_phi, float(delta_t)
                    )

        return restoring(name, steps)

    def mules_correct(name: str, base: str, data: Any) -> np.ndarray:
        delta_t, c_alpha = data

        def step(alpha: Any) -> None:
            alpha_phi_un = phase_flux(name, float(c_alpha))
            corr = alpha_phi_un - nfb.upwind_flux(fields[base], fields["phi"])
            nfb.mules_correct(alpha, alpha_phi_un, corr, float(delta_t))

        return restoring(name, step)

    ops: dict[tuple[str, tuple[str, ...]], Op] = {
        ("field.reload", ("U",)): lambda d: reload_velocity(),
        ("flux.update", ("U",)): lambda d: update_flux(),
        ("vof.compressionVelocity", ("phi",)): compression_velocity,
    }
    for name in _ALPHA_FIELDS:
        ops |= {
            ("field.reload", (name,)): lambda d, name=name: reload_scalar(name),
            ("vof.alphaPhaseFlux", (name,)): lambda d, name=name: _to_numpy(
                (composed_phase_flux if d[1] else phase_flux)(name, float(d[0])).internal_vector()
            ),
            ("vof.linearCompressionFlux", (name,)): lambda d, name=name: linear_compression_flux(
                name, d
            ),
            ("mules.explicitSolve", (name,)): lambda d, name=name: mules_explicit(name, d),
            ("vof.courantNumbers", (name,)): lambda d, name=name: np.array(
                nfb.courant_numbers(fields[name], fields["phi"], float(d[0]))
            ),
        }
        for base in _ALPHA_FIELDS:
            ops[("mules.correct", (name, base))] = lambda d, name=name, base=base: mules_correct(
                name, base, d
            )
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
    nn.initialize(["neonVofParityWorker"])
    serve(Path(sys.argv[1]).resolve(), sys.argv[2])
    gc.collect()  # drop any cyclic refs holding Kokkos views before finalize
    nn.finalize()


if __name__ == "__main__":
    main()
