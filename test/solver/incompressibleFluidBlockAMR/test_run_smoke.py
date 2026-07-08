# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""C8 — run() executes the bundled periodic/box smoke case end-to-end.

Correctness is deferred (Spec 02): the bar here is "runs to endTime without
NaNs and stays divergence-free after projection".
"""

import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("neon")

from neofoam.solver.incompressibleFluidBlockAMR import run  # noqa: E402


def test_run_smoke_finishes_without_nans(blockamr_session, box_case):
    ctx = run(["incompressibleFluidBlockAMR"])

    engine = ctx.models["blockamr_engine"]

    # Velocity finite everywhere after the full run.
    for a in engine.U.mf[0].arrays():
        assert np.isfinite(np.asarray(a)).all()

    # Pressure finite too.
    for a in engine.p.mf[0].arrays():
        assert np.isfinite(np.asarray(a)).all()

    # Divergence-free flux after the final projection.
    dx = engine.mesh.geom(0).cell_size()
    phi = engine.phi
    max_div = 0.0
    face_arrs = [phi[0][d].mf.arrays() for d in range(3)]
    for bi in range(len(face_arrs[0])):
        div_val = None
        for d in range(3):
            f = face_arrs[d][bi][:, :, :, 0]
            ng = phi[0][d].mf.n_grow()
            nc = [int(f.shape[ax]) - 2 * ng - (1 if ax == d else 0) for ax in range(3)]
            hi = [slice(ng, ng + nc[ax]) for ax in range(3)]
            lo = [slice(ng, ng + nc[ax]) for ax in range(3)]
            hi[d] = slice(ng + 1, ng + 1 + nc[d])
            lo[d] = slice(ng, ng + nc[d])
            contrib = (f[tuple(hi)] - f[tuple(lo)]) / dx[d]
            div_val = contrib if div_val is None else div_val + contrib
        max_div = max(max_div, float(jnp.max(jnp.abs(div_val))))
    assert max_div < 1e-6
