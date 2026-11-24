# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""INT-4 / C3 — MeshDictConfig builds a blockamr mesh with correct geometry."""

import pytest

pytest.importorskip("neon")

from neofoam.solver.incompressibleFluidBlockAMR.configs import MeshDictConfig  # noqa: E402
from neofoam.solver.incompressibleFluidBlockAMR.models.mesh_factory import (  # noqa: E402
    build_mesh,
)


def test_extents_cellsize_periodicity(blockamr_session):
    cfg = MeshDictConfig(
        domain=[[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]],
        nCell=[8, 16, 4],
        periodicity=[True, False, True],
    )
    mesh = build_mesh(cfg)
    geom = mesh.geom(0)

    assert list(geom.prob_lo()) == pytest.approx([0.0, 0.0, 0.0])
    assert list(geom.prob_hi()) == pytest.approx([2.0, 4.0, 6.0])
    assert list(geom.cell_size()) == pytest.approx([2.0 / 8, 4.0 / 16, 6.0 / 4])
    assert list(geom.is_periodic()) == [1, 0, 1]
    assert mesh.n_levels() == 1


def test_single_level_when_no_refinement(blockamr_session):
    cfg = MeshDictConfig(
        domain=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        nCell=[8, 8, 8],
        periodicity=[True, True, True],
    )
    mesh = build_mesh(cfg)
    assert mesh.max_level == 0
