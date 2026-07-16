# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""M3 end-to-end — run() drives the config-driven cylinder case.

Exercises the whole non-periodic + immersed-body path through the framework: a
``system/meshDict`` with an ``eb { type cylinder; ... }`` block and a per-face
``boundary`` dict (inlet fixedValue, outlet zeroGradient, slip walls). The bar is
"runs to endTime without NaNs, stays bounded, keeps the body no-slip". Literature
drag/observables are Spec 03.
"""

import numpy as np
import pytest

pytest.importorskip("neon")

from neofoam.solver.incompressibleFluidBlockAMR import run  # noqa: E402


def test_cylinder_case_runs_and_enforces_body(blockamr_session, cylinder_case):
    from blockamr.ibm import DirectForcing

    ctx = run(["incompressibleFluidBlockAMR"])
    engine = ctx.models["projection_state"]

    # the immersed body was built from the eb config (geometry on mesh.body,
    # direct-forcing masks precomputed on the mesh)
    assert engine.mesh.body is not None
    mask = np.array(engine.mesh.ibm_data(DirectForcing).masks[0][0])
    assert mask.sum() > 0

    arr = np.asarray(engine.U.mf[0].arrays()[0])
    assert np.isfinite(arr).all()
    assert float(np.max(np.abs(arr))) < 5.0  # bounded (U_inf = 1)

    # no-slip body: velocity pinned to ~0 in the solid cells (the mask covers
    # the valid region; ``arrays()`` returns the grown array — slice ghosts off)
    ng = engine.U.mf[0].n_grow()
    nx, ny, nz = mask.shape
    valid = arr[ng : ng + nx, ng : ng + ny, ng : ng + nz, :]
    assert float(np.max(np.abs(valid[mask]))) < 1e-6

    # plotfiles were written
    written = [p.name for p in cylinder_case.iterdir()]
    assert any(name.startswith("plt") or name.startswith("0") for name in written)
