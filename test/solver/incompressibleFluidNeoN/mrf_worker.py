# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The MRF frame operations of one case, on one backend, printed as JSON.

Run as ``python mrf_worker.py <case> {openfoam|neon}``; the answer is the single
stdout line prefixed ``#RESULT`` (OpenFOAM and NeoN both write freely to stdout,
so the result needs a marker), mapping quantity name to a nested list.

A subprocess and not a helper the test imports: ``Foam::Time`` and the Kokkos
runtime are per-process singletons, and the NeoN runtime's ``MeshAdapter``
occupies the same ``region0`` registry slot a pybFoam ``fvMesh`` would, so the
two backends cannot be built in one interpreter. The values come back on stdout
rather than as ``.npy`` artifacts because this probe case is 128 cells — the
``.npy`` convention elsewhere in this suite is for whole-case field dumps.

The two branches are deliberately adjacent: they are the same six operations,
and a reader should be able to see that the comparison is like-for-like.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

RESULT_PREFIX = "#RESULT "


def openfoam(case: Path, patches: list[str]) -> dict[str, Any]:
    """The reference: ``Foam::IOMRFZoneList`` on the OpenFOAM mesh."""
    # Imported per branch, not at module scope: neither backend may load the
    # other's bindings into this interpreter (see the module docstring).
    import pybFoam as pyf  # noqa: PLC0415
    from pybFoam import fvc, surfaceScalarField, volVectorField  # noqa: PLC0415

    # Foam::Time keeps a raw reference to the argList; letting it be collected
    # corrupts every later dictionary read (see create_fields.py).
    args = pyf.argList(["mrf"])
    runtime = pyf.Time(args)
    mesh = pyf.fvMesh(runtime)
    U = volVectorField.read_field(mesh, "U")
    mrf = pyf.IOMRFZoneList(mesh)

    def boundary(field: Any) -> np.ndarray:
        # Patch order, the order the NeoN boundary ranges are laid out in.
        # Copied throughout: the bound accessors hand back a *view* of the Foam
        # ``Field``, which the numpy array would otherwise outlive.
        return np.concatenate([np.asarray(field[patch]).copy() for patch in patches])

    ddt = mrf.DDt(U)  # a tmp: hold it while its internal field is copied
    acceleration = np.asarray(ddt.ref().internalField()).copy()
    filtered = surfaceScalarField(pyf.Word("filtered"), mrf.zeroFilter(fvc.flux(U)))
    phi = surfaceScalarField(pyf.Word("phi"), fvc.flux(U))
    mrf.makeRelative(phi)
    mrf.correctBoundaryVelocity(U)

    return {
        "acceleration": acceleration,
        "zero_filter": np.asarray(filtered.internalField()).copy(),
        "zero_filter_boundary": boundary(filtered),
        "relative_flux": np.asarray(phi.internalField()).copy(),
        "relative_flux_boundary": boundary(phi),
        # The reference for the Python-composed flux is the same makeRelative.
        "composed_relative_flux": np.asarray(phi.internalField()).copy(),
        "composed_relative_flux_boundary": boundary(phi),
        "boundary_velocity": boundary(U),
    }


def neon(case: Path, patches: list[str]) -> dict[str, Any]:
    """The subject: ``nfb.MRFNeoN`` on the NeoN mesh."""
    # As in openfoam(): per-branch so this process loads only one backend.
    import pybFoam as pyf  # noqa: PLC0415

    from neofoam import neofoam_bindings as nfb  # noqa: PLC0415
    from neofoam.solver.neon_runtime import (  # noqa: PLC0415
        ensure_neon_initialized,
        requested_executor,
    )

    ensure_neon_initialized(["mrf"])
    # As above: the argList must outlive the Time, or the fvSchemes conversion
    # inside create_adapter_run_time reads freed memory.
    args = pyf.argList(["mrf"])
    runtime = pyf.Time(args)
    rt = nfb.create_adapter_run_time(runtime, requested_executor())
    rt.fv_schemes_dict = nfb.map_fv_schemes(rt.fv_schemes_dict)

    U = nfb.read_vector_volume_field(rt, "U")
    mrf = nfb.MRFNeoN(rt)

    def host(vector: Any) -> np.ndarray:
        # Copied: ``copy_to_host`` hands back a temporary NeoN Vector.
        return np.asarray(vector.copy_to_host()).copy()

    acceleration = host(mrf.acceleration(U).internal_vector())
    phi = nfb.create_phi(rt, "U")
    filtered = mrf.zero_filter(phi)
    # The field-arithmetic equivalent of make_relative, on the same flux the kernel
    # below consumes: the contribution calls the kernel, and pinning both against
    # OpenFOAM is what proves composing it by hand would agree.
    composed = (phi - mrf.frame_flux) * mrf.relative_keep
    mrf.make_relative(phi)
    mrf.correct_boundary_velocity(U)

    return {
        "acceleration": acceleration,
        "zero_filter": host(filtered.internal_vector()),
        "zero_filter_boundary": host(filtered.boundary_data_value()),
        "relative_flux": host(phi.internal_vector()),
        "relative_flux_boundary": host(phi.boundary_data_value()),
        "composed_relative_flux": host(composed.internal_vector()),
        "composed_relative_flux_boundary": host(composed.boundary_data_value()),
        "boundary_velocity": host(nfb.vector_boundary_values(U)),
    }


def main() -> None:
    case, backend = Path(sys.argv[1]), sys.argv[2]
    patches = sys.argv[3:]
    os.chdir(case)
    quantities = {"openfoam": openfoam, "neon": neon}[backend](case, patches)
    print(RESULT_PREFIX + json.dumps({k: v.tolist() for k, v in quantities.items()}))
    # The answer is complete; leave without running interpreter teardown, which
    # tears the Foam registry (and the Kokkos runtime) down in an order neither
    # backend survives — a crash there would be reported as a failed comparison.
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
