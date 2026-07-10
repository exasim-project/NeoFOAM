# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Evaluate explicit NeoN operators and dump the results as ``.npy``.

Run as a standalone subprocess (``python run_neon_ops.py <case_dir> <out_dir>
<executor> <op> [<op> ...]``) — NeoN/Kokkos and ``Foam::Time`` both hold
per-process global state. Operator keys are defined in
``operator_defs.OPERATORS``; ``executor`` is ``Serial``, ``CPU`` or ``GPU``.

Explicit DSL operators (``nn.exp.*`` / ``nfb.exp_div``) are evaluated through
``nfb.evaluate_explicit(op, schemes, result)``, which zeroes ``result``,
resolves the operator's scheme from the mapped ``fvSchemes`` dictionary (the
same file the pybFoam runner reads) and runs its ``explicitOperation``.
Surface results keep NeoN's layout (internal faces first, boundary faces
appended) — the comparison slices to the OpenFOAM length.
"""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path
from typing import Any

import neon._neon as nn
import numpy as np
import pybFoam as pyf

from neofoam import neofoam_bindings as nfb


def _to_numpy(vec: Any) -> np.ndarray:
    return np.asarray(vec.copy_to_host())


def compute_ops(case_dir: Path, out_dir: Path, executor: str, ops: list[str]) -> None:
    """Evaluate ``ops`` and save results. Kokkos must already be initialized.

    Every NeoN object is local to this function so that it is destroyed before
    ``main`` calls ``nn.finalize()`` — deallocating a Kokkos view after
    finalize aborts the process.
    """
    os.chdir(case_dir)
    arg_list = pyf.argList(["neonOpsRunner"])
    run_time = pyf.Time(arg_list)
    rt = nfb.create_adapter_run_time(run_time, executor)
    schemes = nfb.map_fv_schemes(rt.fv_schemes_dict)

    t_field = nfb.read_scalar_volume_field(rt, "T")
    u_field = nfb.read_vector_volume_field(rt, "U")
    phi = nfb.create_phi(rt, "U")
    gamma = nfb.create_uniform_surface_field(rt, "Gamma", 1.0)
    n_cells = t_field.size()

    def evaluate_scalar(op: Any) -> np.ndarray:
        result = nn.ScalarVector(rt.executor, n_cells, 0.0)
        nfb.evaluate_explicit(op, schemes, result)
        return _to_numpy(result)

    def evaluate_vector(op: Any) -> np.ndarray:
        result = nn.VectorVector(rt.executor, n_cells, nn.Vec3(0.0, 0.0, 0.0))
        nfb.evaluate_explicit(op, schemes, result)
        return _to_numpy(result)

    def interpolate_t() -> np.ndarray:
        interp = nn.SurfaceInterpolationScalar(
            rt.executor, rt.nf_mesh, nn.TokenList(["linear"])
        )
        return _to_numpy(interp.interpolate(t_field).internal_vector())

    results: dict[str, Any] = {
        "interpolate_T": interpolate_t,
        "flux_U": lambda: _to_numpy(nfb.flux(u_field).internal_vector()),
        "grad_T": lambda: evaluate_vector(nn.exp.grad(t_field)),
        "div_phi": lambda: evaluate_scalar(nn.exp.div(phi)),
        "div_phi_T": lambda: evaluate_scalar(nn.exp.div(phi, t_field)),
        "div_phi_U": lambda: evaluate_vector(nfb.exp_div(phi, u_field)),
        "laplacian_Gamma_T": lambda: evaluate_scalar(nn.exp.laplacian(gamma, t_field)),
        "laplacian_Gamma_U": lambda: evaluate_vector(nn.exp.laplacian(gamma, u_field)),
    }

    for op in ops:
        np.save(out_dir / f"{op}.npy", results[op]())


def main() -> None:
    case_dir, out_dir = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    nn.initialize(["neonOpsRunner"])
    compute_ops(case_dir, out_dir, sys.argv[3], sys.argv[4:])
    gc.collect()  # drop any cyclic refs holding Kokkos views before finalize
    nn.finalize()


if __name__ == "__main__":
    main()
