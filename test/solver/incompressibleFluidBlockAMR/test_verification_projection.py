# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""VER-3 — projection identities: discrete div/grad adjoint + divergence-free.

Two algebraic properties the incompressible projection depends on, both checked
through the framework blockAMR operators:

* **Adjoint consistency** — the discrete gradient (``neon.blockamr.dsl.exp.grad``,
  the central-difference ``Grad`` the solver uses) and its induced divergence are
  exact negative adjoints on the periodic domain: ``<div u, phi> = -<u, grad phi>``
  to machine epsilon, independent of resolution. This is summation-by-parts: it is
  what makes the pressure correction ``U -= dt grad(p)`` remove exactly the
  divergence the pressure solve saw.
* **Divergence-free result** — after one ``DSLIncompressibleSolver.step()`` the MAC
  face flux is divergence-free to the pressure-solve tolerance, *regardless* of how
  non-solenoidal the initial velocity was (started here from a random field).
"""

import numpy as np
import pytest

pytest.importorskip("neon")

import jax.numpy as jnp  # noqa: E402
import neon.blockamr as blockamr  # noqa: E402
from neon.blockamr.dsl import exp  # noqa: E402
from neon.blockamr.dsl_solver import DSLIncompressibleSolver  # noqa: E402
from neon.blockamr.field import CellField  # noqa: E402
from neon.blockamr.fillpatch import FillPatchCellConservative  # noqa: E402

from neofoam.solver.incompressibleFluidBlockAMR.configs import (  # noqa: E402
    MeshDictConfig,
)
from neofoam.solver.incompressibleFluidBlockAMR.models.mesh_factory import (  # noqa: E402
    build_mesh,
)


def _unit_cube_mesh(n):
    """Framework config->mesh: single-level periodic [0,1]^3 at N=n per axis."""
    cfg = MeshDictConfig(
        domain=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        nCell=[n, n, n],
        periodicity=[True, True, True],
    )
    return build_mesh(cfg)


def _random_scalar_field(mesh, values):
    """A scalar CellField (ngrow=1) seeded with per-cell ``values`` (N,N,N), ghosts
    filled by the periodic conservative fill patch (single box)."""
    field = CellField(mesh, ncomp=1, ngrow=1, name="s")
    for mfi in blockamr.MFIterator(field.mf[0]):
        field.mf[0].copy_from(mfi, jnp.asarray(values, dtype=float))
    field.fill_patch(0, 0.0)
    return field


def _grad(field):
    """Central-difference gradient of a scalar field as a valid-shaped (N,N,N,3)
    numpy array (single box) — the framework ``exp.grad`` operator."""
    grad_op = exp.grad(field)
    result = None
    for mfi in blockamr.MFIterator(field.mf[0]):
        arr = jnp.asarray(field.mf[0].grown_array(mfi)[:, :, :, 0])
        result = np.asarray(grad_op.build_kernel(mfi, t=0.0)(arr))
    return result


def test_grad_div_adjoint_identity_machine_epsilon(blockamr_session):
    """<div u, phi> == -<u, grad phi> to machine epsilon (periodic, arbitrary u/phi).

    ``div u`` is assembled from the same ``exp.grad`` operator (trace of grad(u)),
    so this verifies the discrete gradient is its own negative adjoint under the
    periodic inner product weighted by the cell volume.
    """
    n = 24
    rng = np.random.default_rng(0)
    mesh = _unit_cube_mesh(n)
    dx = mesh.geom(0).cell_size()
    cell_vol = float(dx[0] * dx[1] * dx[2])

    phi_vals = rng.standard_normal((n, n, n))
    u_vals = [rng.standard_normal((n, n, n)) for _ in range(3)]

    phi = _random_scalar_field(mesh, phi_vals)
    u_fields = [_random_scalar_field(mesh, u_vals[d]) for d in range(3)]

    grad_phi = _grad(phi)  # (N,N,N,3)
    # div u = d(u_x)/dx + d(u_y)/dy + d(u_z)/dz, each component from exp.grad.
    div_u = sum(_grad(u_fields[d])[..., d] for d in range(3))

    inner_div = float(np.sum(div_u * phi_vals) * cell_vol)
    inner_grad = float(
        np.sum(sum(u_vals[d] * grad_phi[..., d] for d in range(3))) * cell_vol
    )

    scale = abs(inner_div) + abs(inner_grad) + 1.0
    residual = abs(inner_div + inner_grad) / scale
    assert residual < 1e-10, (
        f"adjoint residual {residual:.3e}: <div u,phi>={inner_div:.6e}, "
        f"<u,grad phi>={inner_grad:.6e}"
    )


def test_grad_div_adjoint_is_resolution_independent(blockamr_session):
    """The adjoint identity holds at machine epsilon on every grid, not asymptotically."""
    rng = np.random.default_rng(7)
    for n in (16, 32):
        mesh = _unit_cube_mesh(n)
        dx = mesh.geom(0).cell_size()
        cell_vol = float(dx[0] * dx[1] * dx[2])

        phi_vals = rng.standard_normal((n, n, n))
        u_vals = [rng.standard_normal((n, n, n)) for _ in range(3)]
        phi = _random_scalar_field(mesh, phi_vals)
        u_fields = [_random_scalar_field(mesh, u_vals[d]) for d in range(3)]

        grad_phi = _grad(phi)
        div_u = sum(_grad(u_fields[d])[..., d] for d in range(3))
        inner_div = float(np.sum(div_u * phi_vals) * cell_vol)
        inner_grad = float(
            np.sum(sum(u_vals[d] * grad_phi[..., d] for d in range(3))) * cell_vol
        )

        scale = abs(inner_div) + abs(inner_grad) + 1.0
        assert abs(inner_div + inner_grad) / scale < 1e-10, f"N={n}"


# --- divergence-free after projection ---------------------------------------


def _max_face_divergence(phi, mesh):
    """max|div(phi)| over the domain from the MAC face-flux field."""
    dx = mesh.geom(0).cell_size()
    max_abs = 0.0
    face_arrs = [phi[0][d].mf.arrays() for d in range(3)]
    for bi in range(len(face_arrs[0])):
        div_val = None
        for d in range(3):
            f = face_arrs[d][bi][:, :, :, 0]
            ng = phi[0][d].mf.n_grow()
            nc = [int(f.shape[ax]) - 2 * ng - (1 if ax == d else 0) for ax in range(3)]
            sl_hi = [slice(ng, ng + nc[ax]) for ax in range(3)]
            sl_lo = [slice(ng, ng + nc[ax]) for ax in range(3)]
            sl_hi[d] = slice(ng + 1, ng + 1 + nc[d])
            sl_lo[d] = slice(ng, ng + nc[d])
            contrib = (f[tuple(sl_hi)] - f[tuple(sl_lo)]) / dx[d]
            div_val = contrib if div_val is None else div_val + contrib
        max_abs = max(max_abs, float(jnp.max(jnp.abs(div_val))))
    return max_abs


def _set_random_velocity(solver, mesh, seed):
    """Seed U with a bounded random (non-solenoidal) field on the periodic box."""
    rng = np.random.default_rng(seed)
    mf = solver.U.mf[0]
    for mfi in blockamr.MFIterator(mf):
        bx = mfi.valid_box()
        lo, hi = bx.small_end(), bx.big_end()
        shape = tuple(hi[i] - lo[i] + 1 for i in range(3)) + (3,)
        mf.copy_from(mfi, jnp.asarray(rng.standard_normal(shape), dtype=float))
    solver.U.fill_patch(0, 0.0)


def test_projection_makes_velocity_divergence_free(blockamr_session):
    """A random (divergent) initial velocity is divergence-free after one step()."""
    n, nz = 16, 4
    mesh = build_mesh(
        MeshDictConfig(
            domain=[[0.0, 0.0, 0.0], [1.0, 1.0, float(nz) / n]],
            nCell=[n, n, nz],
            periodicity=[True, True, True],
        )
    )
    solver = DSLIncompressibleSolver(
        mesh,
        0.01,
        0.2 / n,
        fill_patch=FillPatchCellConservative(),
        sol_p={"rtol": 1e-12, "atol": 1e-14, "maxIter": 400, "verbose": 0},
    )
    _set_random_velocity(solver, mesh, seed=3)

    solver.step()

    div_after = _max_face_divergence(solver.phi, mesh)
    assert div_after < 1e-6, f"projection left max|div phi| = {div_after:.3e}"
