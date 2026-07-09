# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""INT-7 — lid-driven cavity through the framework solver vs Ghia (Spec 03, P1).

Drives the *config-driven* cavity case (``cases/cavity``: a 16³ unit cube, no-slip
walls + a moving lid, Re=100) end-to-end through
``run(["incompressibleFluidBlockAMR"])`` and checks the vertical-centreline
u-velocity against Ghia et al. (1982), Table I. This is the framework-path analogue
of the engine's ``test_dsl_lid_cavity`` — it proves the *whole* solver stack
(config → mesh → BC mapping → projection → time loop) reproduces the benchmark, not
just the bare engine.

Tolerance is loose (coarse 16-cell grid) and matches the engine test's philosophy:
a band that a correct solver clears but a broken one fails.
"""

import numpy as np
import pytest

pytest.importorskip("neon")

from neofoam.solver.incompressibleFluidBlockAMR import postpro, run  # noqa: E402

# Ghia et al. (1982), Table I — Re=100, u along the vertical centreline x=0.5.
GHIA_Y = np.array(
    [
        0.0000,
        0.0547,
        0.0625,
        0.0703,
        0.1016,
        0.1719,
        0.2813,
        0.4531,
        0.5000,
        0.6172,
        0.7344,
        0.8516,
        0.9531,
        0.9609,
        0.9688,
        0.9766,
        1.0000,
    ]
)
GHIA_U = np.array(
    [
        0.00000,
        -0.03717,
        -0.04192,
        -0.04775,
        -0.06434,
        -0.10150,
        -0.15662,
        -0.21090,
        -0.20581,
        -0.13641,
        0.00332,
        0.23151,
        0.68717,
        0.73722,
        0.78871,
        0.84123,
        1.00000,
    ]
)


def test_cavity_re100_centreline_matches_ghia(blockamr_session, cavity_case):
    """Framework lid-cavity at Re=100 reproduces the Ghia centreline profile."""
    ctx = run(["incompressibleFluidBlockAMR"])
    engine = ctx.models["blockamr_engine"]

    snap = postpro.gather_field(engine)
    ix = snap.u.shape[0] // 2  # x = 0.5 column
    iz = snap.u.shape[2] // 2  # z mid-plane
    u_profile = snap.u[ix, :, iz]
    y = snap.y

    assert np.isfinite(u_profile).all()
    assert float(np.max(np.abs(u_profile))) < 2.0  # bounded (lid U=1 + corner)

    ghia_interp = np.interp(y, GHIA_Y, GHIA_U)

    # relative error where the reference is non-trivial
    mask = np.abs(ghia_interp) > 0.05
    rel_err = np.abs(u_profile[mask] - ghia_interp[mask]) / np.abs(ghia_interp[mask])
    assert float(np.max(rel_err)) < 0.5, "centreline u deviates > 50% from Ghia"

    # absolute error everywhere (catches wrong sign / dead flow)
    assert float(np.max(np.abs(u_profile - ghia_interp))) < 0.3

    # physical structure: reversed flow low in the cavity, forward near the lid
    assert u_profile[y < 0.3].min() < -0.05  # recirculation returns fluid
    assert u_profile[y > 0.9].max() > 0.3  # lid drags fluid forward
