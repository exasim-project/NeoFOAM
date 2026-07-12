# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Solver-core configs for incompressibleFluidBlockAMR.

The block-structured AMReX engine (``neon.blockamr``) is not backed by the
OpenFOAM objectRegistry, so the case is described by a small set of validated
``BaseConfig`` classes bound to plain OpenFOAM dictionaries:

* :class:`ControlDictConfig`  — ``system/controlDict`` time + write control
  (reuses :class:`~neofoam.algorithms.solution_loop.config.TimeControlConfig`).
* :class:`MeshDictConfig`      — ``system/meshDict`` dict-driven Cartesian(+AMR)
  mesh: domain box, cell counts, per-axis periodicity, optional refinement and
  embedded boundary.
* :class:`BlockAMRSolutionConfig` — ``system/fvSolution`` (subdict ``blockAMR``)
  physical viscosity + pressure-Poisson (MLMG) tolerances + advection scheme.

Every field is a scalar / list of scalars so the OpenFOAM reader can parse it
(it does not load dimensioned entries).
"""

import re
from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator

from neofoam.algorithms.solution_loop.config import TimeControlConfig
from neofoam.io import OF, BaseConfig, IOStrategy

# OpenFOAM list tokens survive the file reader as raw strings (the OpenFOAM IO
# strategy stringifies every leaf value), e.g. ``( 16 16 16 )`` or the nested
# ``( ( 0 0 0 ) ( 6.28 6.28 6.28 ) )``. JSON/YAML strategies hand back real
# lists. ``_parse_of_list`` normalises both to nested Python lists of raw string
# atoms; pydantic then coerces the atoms to the field's int/float/bool type.
_OF_TOKEN = re.compile(r"[()]|[^\s()]+")


def _parse_of_list(value: Any) -> Any:
    """Parse an OpenFOAM ``( ... )`` list token into a (possibly nested) list.

    Pass through anything that is not a paren-wrapped string unchanged so
    already-parsed lists (JSON/YAML) and plain scalars are left alone.
    """
    if not isinstance(value, str):
        return value
    s = value.strip()
    if not s.startswith("("):
        return value
    stack: List[List[Any]] = []
    root: Any = value
    for tok in _OF_TOKEN.findall(s):
        if tok == "(":
            new: List[Any] = []
            if stack:
                stack[-1].append(new)
            stack.append(new)
        elif tok == ")":
            root = stack.pop()
        elif stack:
            stack[-1].append(tok)
    return root


@IOStrategy(OF("system/controlDict"))
class ControlDictConfig(TimeControlConfig):
    """``system/controlDict`` — the file-bound time/write config for this solver.

    Inherits the time-stepping + write schema from
    :class:`~neofoam.algorithms.solution_loop.config.TimeControlConfig`. The
    block-structured engine defaults to the CPU executor for tests; ``executor``
    selects CPU vs GPU (the jax/AMReX backend picks it up).
    """

    application: str = "incompressibleFluidBlockAMR"
    executor: str = "cpu"  # cpu | gpu


class RefinementConfig(BaseConfig):
    """Optional AMR block: single-level when ``maxLevel == 0``."""

    maxLevel: int = 0
    refRatio: List[int] = Field(default_factory=lambda: [2, 2, 2])

    _parse_refRatio = field_validator("refRatio", mode="before")(_parse_of_list)


class EmbeddedBoundaryConfig(BaseConfig):
    """Optional embedded boundary. ``none`` for box/periodic cases.

    ``cylinder`` is accepted by the schema but NOT supported by the vendored
    ``neon.blockamr`` engine on this branch (no EB bindings are compiled) — the
    mesh factory raises :class:`NotImplementedError`. Cylinder/EB support is
    deferred to the verification specs (02+).
    """

    type: str = "none"  # none | cylinder
    center: Optional[List[float]] = None
    radius: Optional[float] = None
    axis: Optional[int] = None

    _parse_center = field_validator("center", mode="before")(_parse_of_list)


@IOStrategy(OF("system/meshDict"))
class MeshDictConfig(BaseConfig):
    """``system/meshDict`` — dict-driven Cartesian(+AMR) mesh (no blockMeshDict).

    ``domain`` is the physical ``RealBox`` as ``[[xlo,ylo,zlo],[xhi,yhi,zhi]]``;
    ``nCell`` the coarse-level cell counts; ``periodicity`` the per-axis periodic
    flags. ``refinement`` / ``eb`` are optional.
    """

    domain: List[List[float]]
    nCell: List[int]
    periodicity: List[bool]
    # AMReX ``max_grid_size``: the largest box the domain is chopped into. ``None``
    # keeps the whole domain as a single box (``max(nCell)``); a smaller value
    # splits it into more boxes (more parallel work-units / smaller kernel
    # launches — the knob for the box-size performance sweep).
    maxSize: Optional[int] = None
    # AMReX ``blocking_factor``: rounds ``maxSize`` down to a multiple of it so the
    # split boxes are better coarsenable for the nodal-MLMG projection (some box
    # sizes otherwise stall the cross-box coarse-grid correction). Must divide the
    # ``nCell`` dimensions. ``None`` keeps the plain ``maxSize`` chop.
    blockingFactor: Optional[int] = None
    refinement: RefinementConfig = Field(default_factory=RefinementConfig)

    _parse_lists = field_validator("domain", "nCell", "periodicity", mode="before")(
        _parse_of_list
    )
    eb: EmbeddedBoundaryConfig = Field(default_factory=EmbeddedBoundaryConfig)
    # Per-face velocity BC for non-periodic domains, keyed xlo/xhi/ylo/yhi/zlo/zhi;
    # each entry is an OpenFOAM-style patch spec ({"type": ..., "value": [...]})
    # mapped to a neon.blockamr VectorBC. Empty for fully-periodic cases.
    boundary: Dict[str, Any] = Field(default_factory=dict)


@IOStrategy(OF("system/fvSolution", subdict="blockAMR"))
class BlockAMRSolutionConfig(BaseConfig):
    """``system/fvSolution`` (subdict ``blockAMR``) — solve controls.

    ``nu`` is the kinematic viscosity the engine advects/diffuses with (the
    block-structured engine does not read ``constant/transportProperties``);
    ``rtol`` / ``atol`` / ``maxIter`` configure the pressure-Poisson MLMG solve;
    ``divScheme`` selects the advection scheme (``upwind`` | ``linear`` |
    ``vanLeer`` | ``quick``); ``bottomSolver`` optionally picks the MLMG
    bottom solver (``cg`` | ``bicgstab`` | ``smoother`` | ``cgbicg`` | ``bicgcg``
    | ``default``) — empty leaves AMReX's default (a Krylov solver that converges
    the nodal projection in ~5 V-cycles; ``smoother`` is ~100x slower here).
    """

    nu: float = Field(gt=0.0)
    rtol: float = Field(default=1e-10, gt=0.0)
    atol: float = Field(default=1e-8, gt=0.0)
    maxIter: int = Field(default=200, gt=0)
    divScheme: str = "vanLeer"
    bottomSolver: str = ""
