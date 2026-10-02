# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Element-wise global reductions of a per-group aggregation result."""

from __future__ import annotations

from typing import Any

import numpy as np
import pybFoam as pyf

#: The pybFoam collective behind each reduction, looked up by name at call time
#: so a test can fake it. ``sum`` is ``Foam::gSum``, the global sum of a field.
_COLLECTIVES = {"sum": "sum", "max": "gMax", "min": "gMin"}


def _is_parallel_run() -> bool:
    """True when this process is one rank of an MPI run."""
    return bool(pyf.Pstream.parRun())


def _reduce_over_ranks(operation: str, values: "np.ndarray[Any, Any]") -> "np.ndarray[Any, Any]":
    """Identity in serial; element-wise over the ranks in a decomposed run.

    pybFoam binds no array-valued reduction, so each element is reduced on its
    own through the collectives it does bind (``gSum``/``gMax``/``gMin``, each
    of which reduces a whole field to one number). The element count comes from
    the binner's spec and not from the local data (R6), so every rank enters the
    same number of collectives in the same order.
    """
    if not _is_parallel_run():
        return values
    reduce_one = getattr(pyf, _COLLECTIVES[operation])
    flat = np.asarray(values, dtype=float).ravel()
    reduced = np.array([reduce_one(pyf.scalarField(np.array([value]))) for value in flat])
    return reduced.reshape(values.shape)


def reduce_sum(values: "np.ndarray[Any, Any]") -> "np.ndarray[Any, Any]":
    """The element-wise sum of ``values`` over all ranks."""
    return _reduce_over_ranks("sum", values)


def reduce_max(values: "np.ndarray[Any, Any]") -> "np.ndarray[Any, Any]":
    """The element-wise maximum of ``values`` over all ranks."""
    return _reduce_over_ranks("max", values)


def reduce_min(values: "np.ndarray[Any, Any]") -> "np.ndarray[Any, Any]":
    """The element-wise minimum of ``values`` over all ranks."""
    return _reduce_over_ranks("min", values)
