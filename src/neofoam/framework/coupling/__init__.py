# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Coupling of two sides of an interface, e.g. two regions of a conjugate heat transfer case.

- :class:`CouplingOp` — the loop that repeats both sides' solution until their
  exchange converges; :meth:`CouplingOp.shared` gives both sides one instance.
- :class:`CoupledSide` — what a side sends, how it receives, where its points are.

See :mod:`~neofoam.framework.coupling.loop` for how a boundary condition registers
its side.
"""

from .loop import CoupledSide, CouplingOp, SideChange, identity_transfer, unchanged

__all__ = ["CoupledSide", "CouplingOp", "SideChange", "identity_transfer", "unchanged"]
