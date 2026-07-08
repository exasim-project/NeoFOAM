# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""INT-3 / C4 — staged init builds U/p/phi + the engine into the Context."""

import pytest

pytest.importorskip("neon")

from neofoam.solver.incompressibleFluidBlockAMR.create_fields import create_init  # noqa: E402


def test_staged_init_registers_fields_and_engine(blockamr_session, box_case):
    runner = create_init(box_case)
    ctx = runner.run()

    # Velocity / pressure / flux fields registered.
    assert "U" in ctx.fields
    assert "p" in ctx.fields
    assert "phi" in ctx.fields

    # The DSLIncompressibleSolver engine is in the Context (for the projection op).
    assert "blockamr_engine" in ctx.models
    engine = ctx.models["blockamr_engine"]
    assert engine.U is ctx.fields["U"]
    assert engine.p is ctx.fields["p"]

    # The reused framework core models are present too.
    assert "solution_loop" in ctx.models
    assert "writer" in ctx.models

    # U/p are flagged for writing.
    assert {"U", "p"} <= ctx.write_fields
