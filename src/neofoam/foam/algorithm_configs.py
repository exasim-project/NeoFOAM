# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Typed views of the ``system/fvSolution`` algorithm-control blocks.

The per-spec :class:`~neofoam.foam.fv_configs.fvSolution` slice passes the
``PIMPLE`` / ``PISO`` / ``SIMPLE`` blocks through untyped, so the corrector
counts and switches stayed invisible to ``configurations(solver)`` and the MCP.
Each config here carries the OpenFOAM default of every key its algorithm reads
and is registered on the owning spec so its schema is exported; co-owning
``system/fvSolution`` with the slice is safe because
:func:`neofoam.io.write_configs` merges co-owners under their sub-dict.
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

    ``correctPhi`` is ``Optional`` because OpenFOAM's default is the runtime
    value ``mesh.dynamic()``, not a literal: ``None`` means "not set by the
    case", and :meth:`resolved` substitutes the mesh's answer.
    """

    correctPhi: Optional[bool] = None
    checkMeshCourantNo: bool = False
    moveMeshOuterCorrectors: bool = False

    def resolved(self, *, mesh_dynamic: bool) -> "DynamicMeshControls":
        """A copy whose ``correctPhi`` is the value the algorithm acts on."""
        if self.correctPhi is not None:
            return self
        return self.model_copy(update={"correctPhi": mesh_dynamic})


@IOStrategy(OF("system/fvSolution", subdict="PISO"))
class PisoDynamicMeshControls(DynamicMeshControls):
    """The same mesh-motion switches, read from a ``PISO`` block.

    A twin class because ``IOMetadata.subdict`` is one fixed string per class.
    """


@IOStrategy(OF("system/fvSolution", subdict="PIMPLE"))
class PimpleAlgorithmConfig(BaseConfig):
    """The ``PIMPLE`` block's loop controls, with ``pimpleControl::read``'s defaults.

    ``nCorrectors`` / ``nNonOrthogonalCorrectors`` stay unbounded ``int``s: the
    frozen-flow interIsoFoam tutorials carry deliberate ``-1`` sentinels, and
    the stateful control — not this schema — decides what to reject.
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

    ``nOuterCorrectors`` is pinned to 1 by the loader that selects this class,
    not read: PISO has no outer loop.
    """


@IOStrategy(OF("system/fvSolution", subdict="SIMPLE"))
class SimpleAlgorithmConfig(BaseConfig):
    """The ``SIMPLE`` block's loop controls, with ``simpleControl::read``'s defaults.

    ``consistent yes`` selects SIMPLEC. ``residualControl`` is deliberately not
    modelled — the Python ``SimpleControl`` never reads it, so typing it here
    would advertise a key with no effect; the per-spec ``fvSolution`` slice
    still round-trips it.
    """

    nNonOrthogonalCorrectors: int = 0
    momentumPredictor: bool = True
    consistent: bool = False
