# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Slice 3 — ``backend cpp;`` threads from the case file to the engine.

Proves the vertical wired across plan 05: ``system/fvSolution.solvers.U{backend
cpp}`` → ``USolutionConfig.resolve()`` → ``sol_U["backend"]`` → the ``project``
op's ``UEqn.solve(solution=sol_U)`` → the engine's explicit-backend dispatch
(``solve.py`` ``backends.get(...)``). If the token did not thread, the dispatch
would fall back to ``"jax"`` and ``"cpp"`` would never be selected.
"""

import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("neon")

import blockamr.backends as backends  # noqa: E402

from neofoam.solver.incompressibleFluidBlockAMR import run  # noqa: E402


def _flat_velocity(ctx):
    """Concatenate the projected level-0 velocity into one flat array."""
    return np.concatenate([np.asarray(a).ravel() for a in ctx.fields["U"].mf[0].arrays()])


def _max_divergence(ctx):
    """Discrete divergence of the projected face flux (mirrors the smoke test)."""
    dx = ctx.fields["U"].mesh.geom(0).cell_size()
    phi = ctx.fields["phi"]
    face_arrs = [phi[0][d].mf.arrays() for d in range(3)]
    max_div = 0.0
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
    return max_div


def test_backend_cpp_threads_from_case_file_to_engine(blockamr_session, box_cpp_case, monkeypatch):
    dispatched = []
    real_get = backends.get

    def spy(name):
        dispatched.append(name)
        return real_get(name)

    monkeypatch.setattr(backends, "get", spy)

    ctx = run(["incompressibleFluidBlockAMR"])

    # 1. The case-file token reached the engine: the only explicit backend
    #    selected across the whole run (the momentum predictor, once per step)
    #    is cpp — nothing silently fell back to jax.
    assert dispatched, "no explicit backend was dispatched during the run"
    assert set(dispatched) == {"cpp"}

    # 2. Physically sane: finite everywhere and divergence-free after projection.
    u_cpp = _flat_velocity(ctx)
    assert np.isfinite(u_cpp).all()
    assert _max_divergence(ctx) < 1e-6

    # 3. Physical correctness vs the jax path. deltaT is a negative power of two,
    #    so the jax f32 dt/coeff cast is exact and the two explicit backends
    #    differ only by float64 summation order (accumulated over the steps and
    #    the shared MLMG pressure projection). Flip the token to jax and rerun
    #    the identical case; the projected velocities must match tightly.
    fvsol = box_cpp_case / "system" / "fvSolution"
    fvsol.write_text(fvsol.read_text().replace("backend     cpp;", "backend     jax;"))

    ctx_jax = run(["incompressibleFluidBlockAMR"])
    u_jax = _flat_velocity(ctx_jax)

    np.testing.assert_allclose(u_cpp, u_jax, rtol=1e-9, atol=1e-12)
