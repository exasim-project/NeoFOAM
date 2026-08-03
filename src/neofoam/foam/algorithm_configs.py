# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Typed views of the ``system/fvSolution`` algorithm-control blocks.

``fvSolution``'s ``PIMPLE`` / ``PISO`` / ``SIMPLE`` blocks carry the loop
controls a pressure-velocity algorithm reads at construction. The per-spec
:class:`~neofoam.foam.fv_configs.fvSolution` slice models the ``solvers``
entries an operation declares (plus the optional ``pRefCell`` / ``pRefValue``
control keys) and passes the rest of the block through untyped, so the corrector
counts and switches were invisible to ``configurations(solver)`` and to the MCP.

The configs here close that gap: each is bound to its block via
``@IOStrategy(OF("system/fvSolution", subdict=...))``, carries the OpenFOAM
default of every key the algorithm reads, and is registered on the owning spec
so its schema is exported. They co-own ``system/fvSolution`` with the per-spec
slice — :func:`neofoam.io.write_configs` merges co-owners of a file (and nests
each one under its declared sub-dict), so declaring both does not make them
clobber each other.

The stateful loop objects (:class:`~neofoam.algorithms.solution_loop.control.PimpleControl`
/ :class:`~neofoam.algorithms.solution_loop.control.SimpleControl`) are
unchanged: a solver's control factory loads the config and feeds their
construction.
"""

from __future__ import annotations

from typing import Optional

from neofoam.io import OF, BaseConfig, IOStrategy

__all__ = [
    "DynamicMeshControls",
    "PimpleAlgorithmConfig",
    "PisoAlgorithmConfig",
    "PisoDynamicMeshControls",
    "SimpleAlgorithmConfig",
]


@IOStrategy(OF("system/fvSolution", subdict="PIMPLE"))
class DynamicMeshControls(BaseConfig):
    """The mesh-motion switches of the ``PIMPLE`` block (``createDyMControls.H``).

    ``correctPhi`` is deliberately ``Optional``: OpenFOAM's default is not a
    literal but the runtime value ``mesh.dynamic()`` — a moving mesh projects the
    mapped flux, a static one has nothing to project. ``None`` therefore means
    "not set by the case", and :meth:`resolved` substitutes the mesh's answer at
    the point where a mesh is in hand.
    """

    correctPhi: Optional[bool] = None
    checkMeshCourantNo: bool = False
    moveMeshOuterCorrectors: bool = False

    def resolved(self, *, mesh_dynamic: bool) -> "DynamicMeshControls":
        """A copy whose ``correctPhi`` is the value the algorithm acts on.

        Returns ``self`` when the case set the key; otherwise a copy carrying
        OpenFOAM's ``mesh.dynamic()`` default. Both leave the two literal
        switches untouched.
        """
        if self.correctPhi is not None:
            return self
        return self.model_copy(update={"correctPhi": mesh_dynamic})


@IOStrategy(OF("system/fvSolution", subdict="PISO"))
class PisoDynamicMeshControls(DynamicMeshControls):
    """The same mesh-motion switches, read from a ``PISO`` block.

    A pisoFoam-style case ships ``PISO`` where pimpleFoam ships ``PIMPLE``; the
    algorithm reads its controls from whichever block is present, and
    ``IOMetadata.subdict`` is one fixed string per class — hence the twin.
    """


@IOStrategy(OF("system/fvSolution", subdict="PIMPLE"))
class PimpleAlgorithmConfig(BaseConfig):
    """The ``PIMPLE`` block's loop controls, with ``pimpleControl::read``'s defaults.

    Every default is the one OpenFOAM applies when the key is absent, so a case
    that omits a key keeps the same control it has today. ``nCorrectors`` /
    ``nNonOrthogonalCorrectors`` are plain ``int``s on purpose: the frozen-flow
    interIsoFoam tutorials carry the deliberate ``-1`` sentinels, which the
    stateful control's bounds — not this schema — are responsible for rejecting.
    """

    nOuterCorrectors: int = 1
    nCorrectors: int = 2
    nNonOrthogonalCorrectors: int = 0
    momentumPredictor: bool = True
    turbCorr: bool = True
    turbOnFinalIterOnly: bool = True
    finalOnLastPimpleIterOnly: bool = False


@IOStrategy(OF("system/fvSolution", subdict="PISO"))
class PisoAlgorithmConfig(PimpleAlgorithmConfig):
    """The ``PISO`` block — a single-outer-loop PIMPLE.

    pisoFoam cases ship this block instead of ``PIMPLE``. It is read into the
    same controls, except that ``nOuterCorrectors`` is *pinned* to 1 rather than
    read: PISO has no outer loop. The pin is applied by the loader that selects
    this class (a case that writes ``nOuterCorrectors`` into a ``PISO`` block
    does not get an outer loop out of it, exactly as before).
    """


@IOStrategy(OF("system/fvSolution", subdict="SIMPLE"))
class SimpleAlgorithmConfig(BaseConfig):
    """The ``SIMPLE`` block's loop controls, with ``simpleControl::read``'s defaults.

    ``consistent yes`` selects SIMPLEC. ``residualControl`` is intentionally not
    modelled: the Python ``SimpleControl`` runs one momentum+continuity pass per
    solver step (``useResidualConvergence=False``) and never reads it, so typing
    it here would advertise a key with no effect. It survives a case round-trip
    through the per-spec ``fvSolution`` slice, which passes the rest of the block
    through.
    """

    nNonOrthogonalCorrectors: int = 0
    momentumPredictor: bool = True
    consistent: bool = False
