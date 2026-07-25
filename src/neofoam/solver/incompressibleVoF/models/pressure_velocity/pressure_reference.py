# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Closed-domain pressure reference for the VoF PIMPLE algorithm.

incompressibleVoF always solves for ``p_rgh``; the absolute pressure ``p`` is
reconstructed from it. On a *closed* domain — no boundary patch fixes the
pressure *value* — that reconstruction leaves the pressure level free, and
interFoam's ``pEqn.H`` pins it against ``pRefCell``/``pRefValue``::

    p == p_rgh + rho*gh;
    if (p_rgh.needReference())
    {
        p += dimensionedScalar("p", p.dimensions(),
                               pRefValue - getRefCellValue(p, pRefCell));
        p_rgh = p - rho*gh;
    }

pybFoam binds neither ``GeometricField::needReference()`` nor
``getRefCellValue``, so both are recovered here from bound primitives, in
Python.
"""

import numpy as np
import pybFoam as pyf
from pybFoam import volScalarField

# ``pTraits<scalar>::min`` — what a max-reduction over an empty field returns.
_VGREAT = 1.0e300


def need_reference(ref_cell: int) -> bool:
    """Whether the pressure level is undetermined by the boundary conditions.

    ``Foam::setRefCell`` answers exactly this as its ``bool`` return value, but
    the pybFoam binding drops it. The answer survives in the cell index it does
    hand back, so it is recovered from there rather than re-derived.

    Why the sentinel is trustworthy (``findRefCell.C``, OpenFOAM v2406): a cell
    index is assigned *only* inside ``if (fieldRef.needReference() ||
    forceReference)``, and the ``else`` of that ``if`` — at function scope,
    lines 111-114 — sets ``refCelli = -1``. The pybFoam lambda seeds
    ``label pRefCell = 0`` before the call, but OpenFOAM overwrites it with
    ``-1`` on the no-reference path, so a negative index means "no reference
    needed" and nothing else.

    The two-field overload ``setRefCell(p, p_rgh, dict, …)`` tests
    ``needReference()`` on its *second* argument, which is ``p_rgh`` — the
    field NeoFOAM solves for, and exactly the one interFoam's ``pEqn.H`` guards
    the level shift with.

    Caveat: the sentinel tracks ``needReference()`` only because the call site
    always passes ``forceReference=False``. With ``forceReference=True`` the
    branch is taken regardless, a cell index is always assigned, and a negative
    index would no longer say anything about the boundary conditions.

    Parallel: inside the branch only the rank holding the cell keeps a
    non-negative index (``refCelli = -1`` on the others, and with ``pRefPoint``
    exactly one rank finds the cell), so the per-rank answers are OR-reduced —
    OpenFOAM's ``returnReduce(…, orOp)``.
    """
    return _reduce_or(ref_cell >= 0)


def _reduce_or(flag: bool) -> bool:
    """``returnReduce(flag, orOp)``, composed from the bound ``gMax`` reduction."""
    return bool(pyf.gMax(pyf.scalarField(np.array([1.0 if flag else 0.0]))) > 0.5)


def get_ref_cell_value(field: volScalarField, ref_cell: int) -> float:
    """Value of ``field`` in ``ref_cell``, reduced over all ranks.

    Python stand-in for ``getRefCellValue``. The reference cell lives on
    exactly one rank (``setRefCell`` leaves ``ref_cell`` negative everywhere
    else), so the owner contributes a one-element field and every other rank an
    empty one; a max-reduction over those picks the owner's value, and reduces
    to ``-VGREAT`` when no rank owns a cell at all — the case OpenFOAM's
    ``returnReduce(0, sumOp)`` answers with 0.
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

    The tail of interFoam's ``pEqn.H``: ``p == p_rgh + rho*gh``, then — only
    when the domain is closed — shift ``p`` so ``p[ref_cell] == ref_value`` and
    re-level ``p_rgh`` from the shifted ``p``. ``p + <float>`` promotes the
    shift to a ``dimensionedScalar`` in ``p``'s own dimensions, which is what
    the C++ line spells out.
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
