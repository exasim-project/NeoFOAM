# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""``MeshDictConfig`` → ``neon.blockamr`` mesh.

Builds a single-level :class:`neon.blockamr.Mesh` (or a multi-level
:class:`neon.blockamr.AmrMesh` when ``refinement.maxLevel > 0``) from the
validated dict config: physical ``RealBox`` extents, coarse cell counts, and
per-axis periodicity map to the AMReX ``Geometry``.

Embedded boundaries (``eb.type = cylinder``) are **not** supported by the
vendored engine on this branch — there are no EB bindings compiled — so the
factory raises :class:`NotImplementedError` for them. Cylinder/EB support is a
verification-spec (02+) concern.
"""

from typing import Any

from ..configs import MeshDictConfig


def _validate(cfg: MeshDictConfig) -> None:
    if len(cfg.domain) != 2 or any(len(p) != 3 for p in cfg.domain):
        raise ValueError(
            "meshDict.domain must be [[xlo,ylo,zlo],[xhi,yhi,zhi]]; "
            f"got {cfg.domain!r}"
        )
    if len(cfg.nCell) != 3:
        raise ValueError(f"meshDict.nCell must have 3 entries; got {cfg.nCell!r}")
    if len(cfg.periodicity) != 3:
        raise ValueError(
            f"meshDict.periodicity must have 3 entries; got {cfg.periodicity!r}"
        )
    if any(n < 1 for n in cfg.nCell):
        raise ValueError(f"meshDict.nCell entries must be >= 1; got {cfg.nCell!r}")


def build_mesh(cfg: MeshDictConfig) -> Any:
    """Construct a ``neon.blockamr`` mesh from a validated :class:`MeshDictConfig`.

    Returns a :class:`neon.blockamr.Mesh` (single level) or
    :class:`neon.blockamr.AmrMesh` (``refinement.maxLevel > 0``).
    """
    import neon.blockamr as blockamr
    from neon.blockamr.mesh import AmrMesh, Mesh

    _validate(cfg)

    if cfg.eb.type == "cylinder":
        raise NotImplementedError(
            "eb.type='cylinder' is not supported: the vendored neon.blockamr "
            "engine on this branch has no embedded-boundary bindings. Use a "
            "box/periodic mesh (eb.type='none'); cylinder/EB is deferred to the "
            "verification specs (02+)."
        )
    if cfg.eb.type != "none":
        raise ValueError(
            f"meshDict.eb.type must be 'none' or 'cylinder'; got {cfg.eb.type!r}"
        )

    lo, hi = cfg.domain
    nx, ny, nz = (int(n) for n in cfg.nCell)
    is_per = [1 if p else 0 for p in cfg.periodicity]

    box = blockamr.Box([0, 0, 0], [nx - 1, ny - 1, nz - 1])
    real_box = blockamr.RealBox(
        [float(lo[0]), float(lo[1]), float(lo[2])],
        [float(hi[0]), float(hi[1]), float(hi[2])],
    )
    geom = blockamr.Geometry(box, real_box, 0, is_per)

    max_size = max(nx, ny, nz)

    if cfg.refinement.maxLevel > 0:
        info = blockamr.AmrInfo()
        info.max_level = int(cfg.refinement.maxLevel)
        ref = cfg.refinement.refRatio[0] if cfg.refinement.refRatio else 2
        for lev in range(cfg.refinement.maxLevel):
            info.set_ref_ratio(lev, int(ref))
        info.set_max_grid_size(0, max_size)
        info.set_blocking_factor(0, 4)
        mesh = AmrMesh(geom, info)
        mesh.init_from_scratch(0.0)
        return mesh

    box_array = blockamr.BoxArray(box)
    box_array.max_size(max_size)
    dist_map = blockamr.DistributionMapping(box_array)
    return Mesh(box_array, dist_map, geom)
