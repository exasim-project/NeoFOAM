# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Closed-domain pressure reference for the VoF PIMPLE algorithm.

Reproduces the tail of interFoam's ``pEqn.H``, which pins the free pressure
level of a closed domain against ``pRefCell``/``pRefValue``. pybFoam binds
neither ``GeometricField::needReference()`` nor ``getRefCellValue``, so both are
recovered here from bound primitives.
"""

import numpy as np
import pybFoam as pyf
from pybFoam import volScalarField

# ``pTraits<scalar>::min`` — what a max-reduction over an empty field returns.
_VGREAT = 1.0e300


def need_reference(ref_cell: int) -> bool:
    """Whether the pressure level is undetermined by the boundary conditions.

    ``Foam::setRefCell`` returns this as a ``bool`` the pybFoam binding drops, so
    it is recovered from the cell index, which ``findRefCell.C`` sets to ``-1``
    on the no-reference path. Valid only for ``forceReference=False`` — otherwise
    an index is always assigned. Only the rank owning the cell sees a
    non-negative index, hence the OR-reduction.
    """
    return _reduce_or(ref_cell >= 0)


def _reduce_or(flag: bool) -> bool:
    """``returnReduce(flag, orOp)``, composed from the bound ``gMax`` reduction."""
    return bool(pyf.gMax(pyf.scalarField(np.array([1.0 if flag else 0.0]))) > 0.5)


def get_ref_cell_value(field: volScalarField, ref_cell: int) -> float:
    """Value of ``field`` in ``ref_cell``, reduced over all ranks.

    Python stand-in for ``getRefCellValue``: the owning rank contributes a
    one-element field and every other rank an empty one, so a max-reduction picks
    the owner's value and yields ``-VGREAT`` when no rank owns a cell.
    """
    values = np.asarray(field.internalField())
    local = values[ref_cell : ref_cell + 1] if ref_cell >= 0 else values[:0]
    reduced = float(pyf.gMax(pyf.scalarField(np.ascontiguousarray(local))))
    return 0.0 if reduced <= -_VGREAT else reduced


def update_absolute_pressure(
    p: volScalarField,
    p_rgh: volScalarField,
    rho: volScalarField,
    gh: volScalarField,
    *,
    ref_cell: int,
    ref_value: float,
    needs_reference: bool,
) -> None:
    """Rebuild ``p`` from ``p_rgh`` and, on a closed domain, fix its level.

    The tail of interFoam's ``pEqn.H``: ``p == p_rgh + rho*gh``, then — only when
    the domain is closed — shift ``p`` so ``p[ref_cell] == ref_value`` and
    re-level ``p_rgh`` from the shifted ``p``.
    """
    p.assign(p_rgh + rho * gh)
    if needs_reference:
        p.assign(p + (ref_value - get_ref_cell_value(p, ref_cell)))
        p_rgh.assign(p - rho * gh)


__all__: list[str] = [
    "get_ref_cell_value",
    "need_reference",
    "update_absolute_pressure",
]
