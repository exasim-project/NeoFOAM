# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Persistent NeoN evaluation worker for one staged case and executor.

Started as ``python neon_worker.py <case_dir> <executor>`` by
``backends.Worker``; owns the one ``Foam::Time`` and the Kokkos runtime of
its process, reads the staged fields once, then serves JSON-line requests
from stdin (``{"func", "args", "scheme", "out"}``), saving each result as
``<out>.npy`` and replying ``#RESULT {...}`` on stdout.

Operator notes:
- explicit ops run through ``nfb.evaluate_explicit`` (zero the result vector,
  ``read`` the scheme, ``explicitOperation``); implicit ops through
  ``nfb.evaluate_implicit``, whose matrix-apply ``A·psi - b`` is
  volume-integrated and divided by the cell volumes here.
- div schemes are per-request: the tokens (e.g. ``Gauss upwind``) are handed
  straight to the operator's ``read`` as a TokenList. Everything else reads
  the staged ``system/fvSchemes`` dictionary.
- all results are copied to host; every NeoN object stays local to
  ``_build_ops``/``serve`` so teardown happens before ``nn.finalize()``.
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
from schemes import div_scheme


def _to_numpy(vec: Any) -> np.ndarray:
    return np.asarray(vec.copy_to_host())


def serve(case_dir: Path, executor: str) -> None:
    """Read the case once, then serve requests.

    Everything lives in this frame for the whole serve loop: ``arg_list`` and
    ``run_time`` must stay referenced (pybFoam holds raw references), and all
    NeoN objects must be destroyed when this frame ends — before ``main``
    finalizes Kokkos.
    """
    os.chdir(case_dir)
    arg_list = pyf.argList(["neonOpsWorker"])
    run_time = pyf.Time(arg_list)
    rt = nfb.create_adapter_run_time(run_time, executor)
    fv_schemes = nfb.map_fv_schemes(rt.fv_schemes_dict)

    t = nfb.read_scalar_volume_field(rt, "T")
    u = nfb.read_vector_volume_field(rt, "U")
    gamma = nfb.create_uniform_surface_field(rt, "Gamma", 1.0)
    phi = nfb.create_phi(rt, "U")
    n_cells = t.size()
    volumes = _to_numpy(rt.nf_mesh.cell_volumes)

    def div_tokens(scheme: str, field: str) -> Any:
        return nn.TokenList(div_scheme(scheme, field).split())

    def explicit_scalar(op: Any, schemes: Any) -> np.ndarray:
        result = nn.ScalarVector(rt.executor, n_cells, 0.0)
        nfb.evaluate_explicit(op, schemes, result)
        return _to_numpy(result)

    def explicit_vector(op: Any, schemes: Any) -> np.ndarray:
        result = nn.VectorVector(rt.executor, n_cells, nn.Vec3(0.0, 0.0, 0.0))
        nfb.evaluate_explicit(op, schemes, result)
        return _to_numpy(result)

    def implicit_scalar(op: Any, psi: Any, schemes: Any) -> np.ndarray:
        result = nn.ScalarVector(rt.executor, n_cells, 0.0)
        nfb.evaluate_implicit(op, schemes, psi, result)
        return _to_numpy(result) / volumes

    def implicit_vector(op: Any, psi: Any, schemes: Any) -> np.ndarray:
        result = nn.VectorVector(rt.executor, n_cells, nn.Vec3(0.0, 0.0, 0.0))
        nfb.evaluate_implicit(op, schemes, psi, result)
        return _to_numpy(result) / volumes[:, None]

    def interpolate_t(_: Any) -> np.ndarray:
        interp = nn.SurfaceInterpolationScalar(
            rt.executor, rt.nf_mesh, nn.TokenList(["linear"])
        )
        return _to_numpy(interp.interpolate(t).internal_vector())

    ops: dict[tuple[str, tuple[str, ...]], Callable[[Any], np.ndarray]] = {
        ("interpolate", ("T",)): interpolate_t,
        ("flux", ("U",)): lambda s: _to_numpy(nfb.flux(u).internal_vector()),
        ("exp.grad", ("T",)): lambda s: explicit_vector(nn.exp.grad(t), fv_schemes),
        ("exp.div", ("phi",)): lambda s: explicit_scalar(nn.exp.div(phi), fv_schemes),
        ("exp.div", ("phi", "T")): lambda s: explicit_scalar(
            nn.exp.div(phi, t), div_tokens(s, "T")
        ),
        ("exp.div", ("phi", "U")): lambda s: explicit_vector(
            nfb.exp_div(phi, u), div_tokens(s, "U")
        ),
        ("exp.laplacian", ("Gamma", "T")): lambda s: explicit_scalar(
            nn.exp.laplacian(gamma, t), fv_schemes
        ),
        ("exp.laplacian", ("Gamma", "U")): lambda s: explicit_vector(
            nn.exp.laplacian(gamma, u), fv_schemes
        ),
        ("imp.div", ("phi", "T")): lambda s: implicit_scalar(
            nn.imp.div(phi, t), t, div_tokens(s, "T")
        ),
        ("imp.div", ("phi", "U")): lambda s: implicit_vector(
            nn.imp.div(phi, u), u, div_tokens(s, "U")
        ),
        ("imp.laplacian", ("Gamma", "T")): lambda s: implicit_scalar(
            nn.imp.laplacian(gamma, t), t, fv_schemes
        ),
        ("imp.laplacian", ("Gamma", "U")): lambda s: implicit_vector(
            nn.imp.laplacian(gamma, u), u, fv_schemes
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
    nn.initialize(["neonOpsWorker"])
    serve(Path(sys.argv[1]).resolve(), sys.argv[2])
    gc.collect()  # drop any cyclic refs holding Kokkos views before finalize
    nn.finalize()


if __name__ == "__main__":
    main()
