# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Operator patch tests on the framework blockAMR path.

Drives the same DSL grad/div/laplacian operators the ``incompressibleFluidBlockAMR``
solver uses (``blockamr.dsl.exp`` / ``.dsl.solve``) on a mesh built through the
framework's config->mesh factory (``build_mesh(MeshDictConfig)``), and checks:
linear/uniform fields are differentiated exactly (machine epsilon), trig fields
converge at 2nd order under grid refinement. Order is measured with the refinement
harness ``observed_order`` shared with the other verification tests.
"""

import math

import numpy as np
import pytest

pytest.importorskip("neon")

import blockamr  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from blockamr.dsl import exp  # noqa: E402
from blockamr.dsl.solve import evaluate  # noqa: E402
from blockamr.field import CellField, FaceField  # noqa: E402
from blockamr.operators.div import Div, update_face_fluxes  # noqa: E402
from blockamr.operators.grad import Grad  # noqa: E402
from blockamr.schemes.div_schemes import Linear  # noqa: E402

from neofoam.solver.incompressibleFluidBlockAMR.configs import (  # noqa: E402
    MeshDictConfig,
)
from neofoam.solver.incompressibleFluidBlockAMR.models.mesh_factory import (  # noqa: E402
    build_mesh,
)

from .verification_helpers import (  # noqa: E402
    observed_order,
)

TWO_PI = 2.0 * math.pi


def _unit_cube_mesh(n):
    """Framework config->mesh: single-level periodic [0,1]^3 at N=n per axis."""
    cfg = MeshDictConfig(
        domain=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        nCell=[n, n, n],
        periodicity=[True, True, True],
    )
    return build_mesh(cfg)


def _cell_centres(lo, shape, dx, prob_lo):
    nx, ny, nz = shape
    xs = np.array([prob_lo[0] + (lo[0] + i + 0.5) * dx[0] for i in range(nx)])
    ys = np.array([prob_lo[1] + (lo[1] + j + 0.5) * dx[1] for j in range(ny)])
    zs = np.array([prob_lo[2] + (lo[2] + k + 0.5) * dx[2] for k in range(nz)])
    return np.meshgrid(xs, ys, zs, indexing="ij")


def _fill_scalar(field, geom, func):
    """Set a scalar CellField to func(X,Y,Z) at cell centres, then fill ghosts."""
    dx = geom.cell_size()
    prob_lo = geom.prob_lo()
    for mfi in blockamr.MFIterator(field.mf[0]):
        bx = mfi.valid_box()
        lo, hi = bx.small_end(), bx.big_end()
        shape = tuple(hi[i] - lo[i] + 1 for i in range(3))
        X, Y, Z = _cell_centres(lo, shape, dx, prob_lo)
        field.mf[0].copy_from(mfi, jnp.asarray(func(X, Y, Z), dtype=float))
    field.fill_patch(0, 0.0)


def _sin3d(X, Y, Z):
    return np.sin(TWO_PI * X) * np.sin(TWO_PI * Y) * np.sin(TWO_PI * Z)


# --- grad --------------------------------------------------------------------


def test_grad_of_linear_field_is_exact(blockamr_session):
    """grad(a x + b y + c z) == (a, b, c) exactly on interior cells."""
    a, b, c = 1.3, -0.7, 2.1
    mesh = _unit_cube_mesh(32)
    geom = mesh.geom(0)
    phi = CellField(mesh, ncomp=1, ngrow=1, name="phi")
    _fill_scalar(phi, geom, lambda X, Y, Z: a * X + b * Y + c * Z)

    grad_op = exp.grad(phi)  # fresh field -> plain central-difference Grad
    assert isinstance(grad_op, Grad)

    max_err = 0.0
    for mfi in blockamr.MFIterator(phi.mf[0]):
        arr = jnp.asarray(phi.mf[0].grown_array(mfi)[:, :, :, 0])
        result = np.asarray(grad_op.build_kernel(mfi, t=0.0)(arr))
        exact = np.zeros_like(result)
        exact[..., 0], exact[..., 1], exact[..., 2] = a, b, c
        # Periodic ghost-wrap corrupts the outermost cell layer (linear field is not
        # periodic); central diff is exact only on interior cells.
        interior = (slice(1, -1), slice(1, -1), slice(1, -1))
        max_err = max(max_err, float(np.max(np.abs(result[interior] - exact[interior]))))
    assert max_err < 1e-10, f"grad(linear) interior error {max_err:.3e}"


def _grad_sin_max_error(n):
    mesh = _unit_cube_mesh(n)
    geom = mesh.geom(0)
    dx, prob_lo = geom.cell_size(), geom.prob_lo()
    phi = CellField(mesh, ncomp=1, ngrow=1, name="phi")
    _fill_scalar(phi, geom, _sin3d)
    grad_op = exp.grad(phi)

    max_err = 0.0
    for mfi in blockamr.MFIterator(phi.mf[0]):
        arr = jnp.asarray(phi.mf[0].grown_array(mfi)[:, :, :, 0])
        result = np.asarray(grad_op.build_kernel(mfi, t=0.0)(arr))
        lo = mfi.valid_box().small_end()
        X, Y, Z = _cell_centres(lo, result.shape[:3], dx, prob_lo)
        sx, sy, sz = np.sin(TWO_PI * X), np.sin(TWO_PI * Y), np.sin(TWO_PI * Z)
        cx, cy, cz = np.cos(TWO_PI * X), np.cos(TWO_PI * Y), np.cos(TWO_PI * Z)
        exact = np.stack(
            [TWO_PI * cx * sy * sz, TWO_PI * sx * cy * sz, TWO_PI * sx * sy * cz],
            axis=-1,
        )
        max_err = max(max_err, float(np.max(np.abs(result - exact))))
    return max_err


def test_grad_of_sine_is_second_order(blockamr_session):
    errors = [_grad_sin_max_error(n) for n in (16, 32, 64)]
    order = observed_order(errors)
    assert order > 1.8, f"grad order {order:.3f}, errs={errors}"
    assert errors[0] / errors[1] > 3.5 and errors[1] / errors[2] > 3.5


# --- laplacian ---------------------------------------------------------------


def _lap_sin_max_error(n):
    mesh = _unit_cube_mesh(n)
    geom = mesh.geom(0)
    dx, prob_lo = geom.cell_size(), geom.prob_lo()
    phi = CellField(mesh, ncomp=1, ngrow=1, name="phi")
    _fill_scalar(phi, geom, _sin3d)

    results = evaluate(exp.laplacian(1.0, phi), t=0.0)[0]  # per-box valid arrays
    max_err = 0.0
    for r in results:
        r = np.asarray(r)
        r = r[..., 0] if r.ndim == 4 else r
        # single-box unit cube -> lo=(0,0,0)
        X, Y, Z = _cell_centres((0, 0, 0), r.shape[:3], dx, prob_lo)
        exact = -12.0 * math.pi**2 * _sin3d(X, Y, Z)
        max_err = max(max_err, float(np.max(np.abs(r - exact))))
    return max_err


def test_laplacian_of_sine_is_second_order(blockamr_session):
    errors = [_lap_sin_max_error(n) for n in (16, 32, 64)]
    order = observed_order(errors)
    assert order > 1.8, f"laplacian order {order:.3f}, errs={errors}"


# --- div ---------------------------------------------------------------------


def _x_vel(x, y, z, t):
    return jnp.ones_like(x), jnp.zeros_like(x), jnp.zeros_like(x)


def _div(mesh, phi, scheme):
    ff = FaceField(mesh, ncomp=1, ngrow=scheme.stencil_width)
    update_face_fluxes(ff[0], _x_vel, mesh.geom(0), t=0.0)
    return evaluate(Div(ff, phi, scheme=scheme), t=0.0)[0]


def test_div_of_uniform_field_is_zero(blockamr_session):
    """div(U * phi), U=(1,0,0), phi==1  ->  0 to machine epsilon."""
    mesh = _unit_cube_mesh(32)
    phi = CellField(mesh, ncomp=1, ngrow=Linear().stencil_width, name="phi")
    _fill_scalar(phi, mesh.geom(0), lambda X, Y, Z: np.ones_like(X))
    for r in _div(mesh, phi, Linear()):
        assert np.max(np.abs(np.asarray(r))) < 1e-12


def _div_sin_max_error(n):
    mesh = _unit_cube_mesh(n)
    geom = mesh.geom(0)
    dx, prob_lo = geom.cell_size(), geom.prob_lo()
    phi = CellField(mesh, ncomp=1, ngrow=Linear().stencil_width, name="phi")
    _fill_scalar(phi, geom, _sin3d)

    max_err = 0.0
    for r in _div(mesh, phi, Linear()):
        r = np.asarray(r)
        r = r[..., 0] if r.ndim == 4 else r
        X, Y, Z = _cell_centres((0, 0, 0), r.shape[:3], dx, prob_lo)
        exact = TWO_PI * np.cos(TWO_PI * X) * np.sin(TWO_PI * Y) * np.sin(TWO_PI * Z)
        max_err = max(max_err, float(np.max(np.abs(r - exact))))
    return max_err


def test_div_of_sine_is_second_order(blockamr_session):
    errors = [_div_sin_max_error(n) for n in (16, 32, 64)]
    order = observed_order(errors)
    assert order > 1.8, f"div order {order:.3f}, errs={errors}"
