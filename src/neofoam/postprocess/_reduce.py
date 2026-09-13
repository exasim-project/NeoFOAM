# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Element-wise global reductions of a per-group aggregation result."""

from __future__ import annotations

from typing import Any

import numpy as np
import pybFoam as pyf


def _is_parallel_run() -> bool:
    """True when this process is one rank of an MPI run."""
    return bool(pyf.Pstream.parRun())


def _reduce_over_ranks(operation: str, values: "np.ndarray[Any, Any]") -> "np.ndarray[Any, Any]":
    """Identity in serial; in parallel there is no binding to call yet (R1)."""
    if not _is_parallel_run():
        return values
    raise NotImplementedError(
        f"postProcess: an element-wise global {operation} over the ranks is not bound in "
        "pybFoam yet (Pstream exposes master/parRun/myProcNo/nProcs, and gMax/gMin reduce a "
        "whole scalarField to one number), so aggregation cannot run decomposed — "
        "run the case serially"
    )


def reduce_sum(values: "np.ndarray[Any, Any]") -> "np.ndarray[Any, Any]":
    """The element-wise sum of ``values`` over all ranks."""
    return _reduce_over_ranks("sum", values)


def reduce_max(values: "np.ndarray[Any, Any]") -> "np.ndarray[Any, Any]":
    """The element-wise maximum of ``values`` over all ranks."""
    return _reduce_over_ranks("max", values)


def reduce_min(values: "np.ndarray[Any, Any]") -> "np.ndarray[Any, Any]":
    """The element-wise minimum of ``values`` over all ranks."""
    return _reduce_over_ranks("min", values)
