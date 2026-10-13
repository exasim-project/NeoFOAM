# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Coupling of two sides of an interface, e.g. two regions of a conjugate heat transfer case.

- :class:`CouplingOp` — the loop that repeats both sides' solution until their
  exchange converges; :meth:`CouplingOp.shared` gives both sides one instance.
- :class:`CoupledSide` — what a side sends, how it receives, where its points are.
- numpy kernels to plug into a coupling: :func:`nearest_neighbor` mapping
  (:class:`MappedTransfer`), :class:`ConstantRelaxation` and
  :class:`AitkenRelaxation`, :class:`RelativeConvergence`.

See :mod:`~neofoam.framework.coupling.loop` for how a boundary condition registers
its side.
"""

from .kernels import (
    AitkenRelaxation,
    ConstantRelaxation,
    MappedTransfer,
    RelativeConvergence,
    nearest_neighbor,
)
from .loop import CoupledSide, CouplingOp, SideChange, identity_transfer, unchanged

__all__ = [
    "AitkenRelaxation",
    "ConstantRelaxation",
    "CoupledSide",
    "CouplingOp",
    "MappedTransfer",
    "RelativeConvergence",
    "SideChange",
    "identity_transfer",
    "nearest_neighbor",
    "unchanged",
]
