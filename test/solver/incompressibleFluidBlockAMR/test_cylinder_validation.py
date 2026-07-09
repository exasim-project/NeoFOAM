# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""E2E-1 — blockAMR cylinder Re=20 observables vs literature (Spec 03, P4/P7).

This is the *validation* tier: it drives the config-driven cylinder case
(``cases/cylinder_re20``) end-to-end through ``run(["incompressibleFluidBlockAMR"])``
to a steady Re=20 wake and extracts the observables from ``postpro`` — the same
math locked by the fast INT-8 canned tests (``test_postpro.py``).

**Opt-in / slow.** A converged run is ~2500 steps at ~0.5-1 s/step (~30-40 min on
one GPU), so it is gated behind ``NEOFOAM_BLOCKAMR_VALIDATION=1`` and skipped in the
normal suite. Run it with::

    NEOFOAM_BLOCKAMR_VALIDATION=1 pytest \
        test/solver/incompressibleFluidBlockAMR/test_cylinder_validation.py -q -s

**On the band (the honest modelling margin).** The literature *unconfined* value is
``Cd ≈ 2.05`` (Tritton; Sucker & Brauer). This case cannot hit that tight band on
this engine because (1) the cylinder is a **direct-forcing staircased** immersed
body (first-order surface, biases Cd high), and (2) the stable geometry is
LX=2 × LY=1 with D=0.2 → **20 % blockage**, which further inflates Cd. Reaching the
tight unconfined band needs the deferred **cut-cell EB** body (exact surface forces)
and a low-blockage domain — see the Spec 03 / EB-next-iteration plan. So this test
asserts the *achievable* band (literature value as the lower edge, widened upward by
the stated staircase+blockage margin) plus the qualitative wake structure; it proves
the observable pipeline runs end-to-end and lands in the physically-correct regime,
not that the staircase body is quantitatively converged.
"""

import os
import shutil

import numpy as np
import pytest

pytest.importorskip("neon")

from neofoam.solver.incompressibleFluidBlockAMR import postpro, run  # noqa: E402

CASE_SRC = "cylinder_re20"  # under cases/, copied by the fixture below

# Re=20 references (unconfined literature) and the achievable modelling band.
CD_LIT = 2.05  # unconfined literature Cd @ Re=20
CD_LO, CD_HI = (
    2.0,
    3.6,
)  # achievable band: lit value floor + staircase/20%-blockage margin
LR_D_LIT = 0.93  # unconfined recirculation length / D


_VALIDATION = os.environ.get("NEOFOAM_BLOCKAMR_VALIDATION") == "1"


@pytest.fixture
def cylinder_re20_case(tmp_path, monkeypatch):
    """Copy the Re=20 validation case to a tmp dir and chdir into it."""
    src = os.path.join(os.path.dirname(__file__), "cases", CASE_SRC)
    dst = tmp_path / CASE_SRC
    shutil.copytree(src, dst)
    monkeypatch.chdir(dst)
    return dst


@pytest.mark.skipif(
    not _VALIDATION,
    reason="slow (~30-40 min GPU) validation run; set NEOFOAM_BLOCKAMR_VALIDATION=1",
)
def test_cylinder_re20_observables_in_band(blockamr_session, cylinder_re20_case):
    """Config-driven Re=20 cylinder: steady, sane wake, Cd in the achievable band."""
    ctx = run(["incompressibleFluidBlockAMR"])
    engine = ctx.models["blockamr_engine"]

    U_inf, D = 1.0, 0.2
    Cd, Cl = postpro.force_coefficients(engine, U_inf, D, nu=0.01, tail_fraction=0.3)
    snap = postpro.gather_field(engine)
    Lr = postpro.recirculation_length(snap, D)
    theta = postpro.separation_angle(snap)

    # field is bounded & finite (stable CFL-0.1 run, no divergence)
    arr = np.asarray(engine.U.mf[0].arrays()[0])
    assert np.isfinite(arr).all()
    assert float(np.max(np.abs(arr))) < 3.0

    # steady flow: negligible lift, positive drag
    assert abs(Cl) < 0.1, f"Cl should be ~0 for steady Re=20, got {Cl}"
    assert Cd > 0, f"drag must be positive, got {Cd}"

    # drag in the achievable band (literature 2.05 + stated staircase/blockage margin)
    assert CD_LO <= Cd <= CD_HI, (
        f"Cd={Cd:.3f} outside achievable band [{CD_LO}, {CD_HI}] "
        f"(unconfined literature {CD_LIT})"
    )

    # a real recirculation bubble forms behind the body (physical structure)
    assert Lr > 0.0, f"no recirculation bubble detected (Lr/D={Lr})"
    assert 0.0 < theta < 180.0, f"separation angle out of range: {theta}"
