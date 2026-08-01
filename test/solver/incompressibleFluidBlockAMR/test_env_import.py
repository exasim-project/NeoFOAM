# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""INT-1 / C1 — the environment gate.

``blockamr`` + ``jax`` import in the neofoam venv, and a small periodic
projection state constructs in-process.
"""

import pytest

pytest.importorskip("neon")

import blockamr  # noqa: E402
from blockamr.fillpatch import FillPatchCellConservative  # noqa: E402
from blockamr.incompressible import build_incompressible  # noqa: E402
from blockamr.mesh import Mesh  # noqa: E402


def test_jax_imports():
    import jax  # noqa: PLC0415 — the import under test

    assert jax.__version__


def test_construct_periodic_solver(blockamr_session):
    """Build an 8^3 fully-periodic solver — no exception, fields present."""
    n = 8
    box = blockamr.Box([0, 0, 0], [n - 1, n - 1, n - 1])
    real_box = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, real_box, 0, [1, 1, 1])
    box_array = blockamr.BoxArray(box)
    box_array.max_size(n)
    dist_map = blockamr.DistributionMapping(box_array)
    mesh = Mesh(box_array, dist_map, geom)

    solver = build_incompressible(
        mesh,
        nu=0.01,
        dt=0.01,
        fill_patch=FillPatchCellConservative(),
    )

    assert solver is not None
    assert solver.U is not None
    assert solver.p is not None
    assert solver.phi is not None
