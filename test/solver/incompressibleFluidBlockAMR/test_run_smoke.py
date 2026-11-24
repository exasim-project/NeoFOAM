# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""C8 — run() executes the bundled periodic/box smoke case end-to-end.

Correctness is deferred (Spec 02): the bar here is "runs to endTime without
NaNs and stays divergence-free after projection".
"""

import numpy as np
import pytest

pytest.importorskip("neon")

import blockamr  # noqa: E402

from neofoam.solver.incompressibleFluidBlockAMR import run  # noqa: E402
from neofoam.solver.incompressibleFluidBlockAMR.configs import ControlDictConfig  # noqa: E402

# The executor the blockamr build cannot run on (a GPU build has no host path).
_UNAVAILABLE = {"GPU": "cpu", "CPU": "gpu"}


def test_run_smoke_finishes_without_nans(blockamr_session, box_case):
    ctx = run(["incompressibleFluidBlockAMR"])

    U = ctx.fields["U"]

    # Velocity finite everywhere after the full run.
    for a in U.mf[0].arrays():
        assert np.isfinite(np.asarray(a)).all()

    # Pressure finite too.
    for a in ctx.fields["p"].mf[0].arrays():
        assert np.isfinite(np.asarray(a)).all()

    # Divergence-free flux after the final projection.
    dx = U.mesh.geom(0).cell_size()
    phi = ctx.fields["phi"]
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
        max_div = max(max_div, float(np.max(np.abs(div_val))))
    assert max_div < 1e-6


def test_run_rejects_an_executor_the_build_cannot_run(blockamr_session, box_case):
    control = ControlDictConfig.load(case_dir=box_case)
    control.model_copy(update={"executor": _UNAVAILABLE[blockamr.executor()]}).save(
        case_dir=box_case
    )

    with pytest.raises(ValueError, match="controlDict executor"):
        run(["incompressibleFluidBlockAMR"])
