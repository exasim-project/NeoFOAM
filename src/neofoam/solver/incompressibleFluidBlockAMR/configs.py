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
* :class:`FvSchemesConfig`     — ``system/fvSchemes`` discretisation scheme
  names (ddt/div/laplacian/grad), flattened to the DSL ``schemes`` dict.
* :class:`USolutionConfig` / :class:`PSolutionConfig` — ``system/fvSolution``
  per-field ``solvers.U`` / ``solvers.p`` linear-solver + IBM + backend blocks.
* :class:`BlockAMRSolutionConfig` — ``system/fvSolution`` (subdict ``blockAMR``)
  the engine's physical viscosity ``nu``.

Every field is a scalar / list of scalars so the OpenFOAM reader can parse it
(it does not load dimensioned entries).
"""

import re
from typing import Any, Dict, List, Literal, Optional

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


@IOStrategy(OF("system/fvSchemes"))
class FvSchemesConfig(BaseConfig):
    """``system/fvSchemes`` — discretisation scheme names per operator.

    Each sub-block maps an OpenFOAM-style operator key to a scheme name the
    engine's ``SCHEME_REGISTRY`` resolves (``ddt`` → ``Euler`` | ``RK2`` |
    ``RK4``; ``div`` → ``upwind`` | ``linear`` | ``vanLeer`` | ``quick``;
    ``laplacian`` / ``grad`` → ``central``). :meth:`resolve` flattens the
    blocks into the single ``schemes`` dict the DSL consumes.
    """

    ddtSchemes: Dict[str, str] = Field(default_factory=dict)
    divSchemes: Dict[str, str] = Field(default_factory=dict)
    laplacianSchemes: Dict[str, str] = Field(default_factory=dict)
    gradSchemes: Dict[str, str] = Field(default_factory=dict)

    def resolve(self) -> Dict[str, str]:
        """Flatten the sub-blocks into the DSL ``schemes`` name dict.

        A ``default`` key expands to the block's bare operator key
        (``ddtSchemes{default Euler}`` → ``{"ddt": "Euler"}``); any other key
        is kept verbatim (``divSchemes{div(phi,U) vanLeer}`` →
        ``{"div(phi,U)": "vanLeer"}``).
        """
        out: Dict[str, str] = {}
        for op, block in (
            ("ddt", self.ddtSchemes),
            ("div", self.divSchemes),
            ("laplacian", self.laplacianSchemes),
            ("grad", self.gradSchemes),
        ):
            for key, value in block.items():
                out[op if key == "default" else key] = value
        return out


class FieldSolutionConfig(BaseConfig):
    """One field's ``fvSolution.solvers[<field>]`` block.

    ``solver`` / ``rtol`` / ``atol`` / ``maxIter`` configure the field's MLMG
    solve; ``bottomSolver`` optionally picks the MLMG bottom solver
    (``cg`` | ``bicgstab`` | ``smoother`` | ...) — empty leaves AMReX's default
    (a Krylov solver converging the nodal projection in ~5 V-cycles;
    ``smoother`` is ~100x slower here). ``ibm`` selects the field's immersed-
    boundary method (``directForcing``; empty = none). ``backend`` picks the
    per-field kernel implementation (``jax`` | ``cpp``) — distinct from
    ``controlDict.executor`` (``cpu`` | ``gpu``), which selects the *device*.
    ``verbose`` / ``bottomVerbose`` set the AMReX MLMG residual-trace level
    (0 = quiet) for the solve and its bottom solver.
    """

    solver: str = "MLMG"
    rtol: float = Field(default=1e-10, gt=0.0)
    atol: float = Field(default=1e-8, gt=0.0)
    maxIter: int = Field(default=200, gt=0)
    bottomSolver: str = ""
    ibm: str = ""
    backend: Literal["jax", "cpp"] = "jax"
    verbose: int = Field(default=0, ge=0)
    bottomVerbose: int = Field(default=0, ge=0)

    def resolve(self) -> Dict[str, Any]:
        """Return the plain ``solution`` dict with empty-string entries dropped.

        Unset ``bottomSolver`` / ``ibm`` (empty string) are omitted so the DSL
        sees only the keys that were actually configured.
        """
        return {k: v for k, v in self.model_dump().items() if v != ""}


@IOStrategy(OF("system/fvSolution", subdict="solvers.U"))
class USolutionConfig(FieldSolutionConfig):
    """``system/fvSolution`` (subdict ``solvers.U``) — velocity solve block."""


@IOStrategy(OF("system/fvSolution", subdict="solvers.p"))
class PSolutionConfig(FieldSolutionConfig):
    """``system/fvSolution`` (subdict ``solvers.p``) — pressure solve block."""


@IOStrategy(OF("system/fvSolution", subdict="blockAMR"))
class BlockAMRSolutionConfig(BaseConfig):
    """``system/fvSolution`` (subdict ``blockAMR``) — engine physics extras.

    ``nu`` is the kinematic viscosity the engine advects/diffuses with (the
    block-structured engine does not read ``constant/transportProperties``).
    The MLMG tolerances, advection scheme and verbosity live in the per-field
    ``solvers.U`` / ``solvers.p`` blocks and ``fvSchemes``.
    """

    nu: float = Field(gt=0.0)
