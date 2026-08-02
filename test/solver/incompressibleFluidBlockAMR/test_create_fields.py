# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""INT-3 / C4 — staged init builds U/p/phi + the equation inputs into the Context."""

import pytest

pytest.importorskip("neon")

from neofoam.solver.incompressibleFluidBlockAMR.create_fields import create_init  # noqa: E402


def test_staged_init_registers_fields_and_state(blockamr_session, box_case):
    runner = create_init(box_case)
    ctx = runner.run()

    # Velocity / pressure / flux fields registered.
    assert "U" in ctx.fields
    assert "p" in ctx.fields
    assert "phi" in ctx.fields

    # No solver-state object: the fields ARE the Context's, and the operations
    # get their equation inputs as plain models.
    assert "blockamr_engine" not in ctx.models
    assert "projection_state" not in ctx.models
    assert ctx.models["nu"] > 0.0
    assert ctx.models["schemes"]
    assert {"sol_U", "sol_p", "ibm"} <= set(ctx.models)

    # The reused framework core models are present too.
    assert "solution_loop" in ctx.models
    assert "writer" in ctx.models

    # U/p are flagged for writing.
    assert {"U", "p"} <= ctx.write_fields
