# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""INT-4 / C3 — MeshDictConfig builds a blockamr mesh with correct geometry.

Also covers the ``body.type = cylinder`` immersed body: the mesh stays a plain
Cartesian grid (the solid is imposed by direct forcing in the engine), and the
``body`` block is validated. ``eb=`` is a deprecated alias for ``body=``.
"""

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


def test_cylinder_body_builds_cartesian_mesh(blockamr_session):
    """A cylinder body builds a plain Cartesian mesh (the body is direct-forced in
    the engine, not cut into the grid)."""
    cfg = MeshDictConfig(
        domain=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        nCell=[8, 8, 8],
        periodicity=[False, False, False],
        body={"type": "cylinder", "center": [0.5, 0.5, 0.5], "radius": 0.2, "axis": 2},
    )
    mesh = build_mesh(cfg)
    assert mesh.n_levels() == 1
    assert list(mesh.geom(0).cell_size()) == pytest.approx([1.0 / 8] * 3)


def test_eb_alias_still_populates_body(blockamr_session):
    """Deprecated ``eb=`` kwarg still validates and populates ``cfg.body``."""
    cfg = MeshDictConfig(
        domain=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        nCell=[8, 8, 8],
        periodicity=[False, False, False],
        eb={"type": "cylinder", "center": [0.5, 0.5, 0.5], "radius": 0.2, "axis": 2},
    )
    assert cfg.body.type == "cylinder"
    assert cfg.body.center == [0.5, 0.5, 0.5]
    mesh = build_mesh(cfg)
    assert mesh.n_levels() == 1


@pytest.mark.parametrize(
    "body",
    [
        {"type": "wedge"},  # unknown type
        {"type": "cylinder", "center": [0.5, 0.5], "radius": 0.2, "axis": 2},  # 2-vec
        {"type": "cylinder", "center": [0.5, 0.5, 0.5], "radius": -1.0, "axis": 2},
        {"type": "cylinder", "center": [0.5, 0.5, 0.5], "radius": 0.2, "axis": 3},
    ],
)
def test_invalid_body_rejected(blockamr_session, body):
    cfg = MeshDictConfig(
        domain=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        nCell=[8, 8, 8],
        periodicity=[False, False, False],
        body=body,
    )
    with pytest.raises(ValueError):
        build_mesh(cfg)
