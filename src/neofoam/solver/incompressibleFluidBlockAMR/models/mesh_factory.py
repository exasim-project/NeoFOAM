# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""``MeshDictConfig`` → ``neon.blockamr`` mesh.

Builds a single-level :class:`neon.blockamr.Mesh` (or a multi-level
:class:`neon.blockamr.AmrMesh` when ``refinement.maxLevel > 0``) from the
validated dict config: physical ``RealBox`` extents, coarse cell counts, and
per-axis periodicity map to the AMReX ``Geometry``.

Embedded boundaries (``eb.type = cylinder``) are represented by a **direct-forcing
immersed boundary**: the mesh stays a plain Cartesian grid and the engine pins the
velocity to zero in the solid cells each step. The mesh factory validates the
``eb`` block and sets the resulting geometry on ``mesh.body`` (a
:class:`~neon.blockamr.ibm.body.Cylinder`); the IBM *method* (which per-field
solve pins the cells) is a separate ``fvSolution`` choice, wired up in
:mod:`~neofoam.solver.incompressibleFluidBlockAMR.models.projection.chorinProjection`.
"""

from typing import Any

from ..configs import MeshDictConfig


def _validate(cfg: MeshDictConfig) -> None:
    if len(cfg.domain) != 2 or any(len(p) != 3 for p in cfg.domain):
        raise ValueError(
            f"meshDict.domain must be [[xlo,ylo,zlo],[xhi,yhi,zhi]]; got {cfg.domain!r}"
        )
    if len(cfg.nCell) != 3:
        raise ValueError(f"meshDict.nCell must have 3 entries; got {cfg.nCell!r}")
    if len(cfg.periodicity) != 3:
        raise ValueError(
            f"meshDict.periodicity must have 3 entries; got {cfg.periodicity!r}"
        )
    if any(n < 1 for n in cfg.nCell):
        raise ValueError(f"meshDict.nCell entries must be >= 1; got {cfg.nCell!r}")
    if cfg.eb.type not in ("none", "cylinder"):
        raise ValueError(
            f"meshDict.eb.type must be 'none' or 'cylinder'; got {cfg.eb.type!r}"
        )
    if cfg.eb.type == "cylinder":
        if cfg.eb.center is None or len(cfg.eb.center) != 3:
            raise ValueError(
                f"cylinder eb needs a 3-vector 'center'; got {cfg.eb.center!r}"
            )
        if cfg.eb.radius is None or cfg.eb.radius <= 0.0:
            raise ValueError(
                f"cylinder eb needs a positive 'radius'; got {cfg.eb.radius!r}"
            )
        if cfg.eb.axis not in (0, 1, 2):
            raise ValueError(
                f"cylinder eb 'axis' must be 0, 1 or 2; got {cfg.eb.axis!r}"
            )


def build_mesh(cfg: MeshDictConfig) -> Any:
    """Construct a ``neon.blockamr`` mesh from a validated :class:`MeshDictConfig`.

    Returns a :class:`neon.blockamr.Mesh` (single level) or
    :class:`neon.blockamr.AmrMesh` (``refinement.maxLevel > 0``).
    """
    import neon.blockamr as blockamr
    from neon.blockamr.ibm.body import Cylinder
    from neon.blockamr.mesh import AmrMesh, Mesh

    _validate(cfg)

    lo, hi = cfg.domain
    nx, ny, nz = (int(n) for n in cfg.nCell)
    is_per = [1 if p else 0 for p in cfg.periodicity]

    box = blockamr.Box([0, 0, 0], [nx - 1, ny - 1, nz - 1])
    real_box = blockamr.RealBox(
        [float(lo[0]), float(lo[1]), float(lo[2])],
        [float(hi[0]), float(hi[1]), float(hi[2])],
    )
    geom = blockamr.Geometry(box, real_box, 0, is_per)

    # AMReX max_grid_size: a single box spanning the domain by default, or the
    # user-set meshDict ``maxSize`` to chop it into more (smaller) boxes.
    max_size = int(cfg.maxSize) if cfg.maxSize else max(nx, ny, nz)
    blocking_factor = int(cfg.blockingFactor) if cfg.blockingFactor else None

    if cfg.refinement.maxLevel > 0:
        info = blockamr.AmrInfo()
        info.max_level = int(cfg.refinement.maxLevel)
        ref = cfg.refinement.refRatio[0] if cfg.refinement.refRatio else 2
        for lev in range(cfg.refinement.maxLevel):
            info.set_ref_ratio(lev, int(ref))
        info.set_max_grid_size(0, max_size)
        info.set_blocking_factor(0, blocking_factor or 4)
        mesh = AmrMesh(geom, info)
        mesh.init_from_scratch(0.0)
        if cfg.eb.type == "cylinder":
            mesh.body = Cylinder(
                centre=cfg.eb.center, radius=cfg.eb.radius, axis=cfg.eb.axis
            )
        return mesh

    box_array = blockamr.BoxArray(box)
    if blocking_factor is not None:
        # Align every box dimension to ``blocking_factor`` (round ``max_size`` down
        # to a multiple) so the split boxes stay coarsenable for the nodal MLMG.
        # NOTE: box decomposition (max_size < domain) currently fails the nodal
        # projection on a single GPU regardless of alignment — see
        # test_max_size_sweep; blocking_factor is honoured here for the AMR /
        # future multi-box path, not a fix for the single-level split.
        aligned = max(blocking_factor, (max_size // blocking_factor) * blocking_factor)
        box_array.max_size(aligned)
    else:
        box_array.max_size(max_size)
    dist_map = blockamr.DistributionMapping(box_array)
    mesh = Mesh(box_array, dist_map, geom)
    if cfg.eb.type == "cylinder":
        mesh.body = Cylinder(
            centre=cfg.eb.center, radius=cfg.eb.radius, axis=cfg.eb.axis
        )
    return mesh
