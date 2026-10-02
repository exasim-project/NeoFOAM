# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""What a node has to know about an array the kernels do not answer."""

from __future__ import annotations

from typing import Any

import numpy as np


def host(values: Any) -> "np.ndarray[Any, Any]":
    """The values as host numpy; a NeoN vector is copied off its executor first."""
    copy_to_host = getattr(values, "copy_to_host", None)
    if copy_to_host is not None:
        return np.asarray(copy_to_host())
    return np.asarray(values)


def is_vector(values: Any) -> bool:
    """Whether the values are 3-vectors rather than scalars."""
    if isinstance(values, np.ndarray):
        return values.ndim == 2
    import neon  # noqa: PLC0415  # NeoN is only needed on the NeoN path

    return isinstance(values, neon.VectorVector)


def ones_like(values: Any) -> Any:
    """One scalar per element of *values*, of the same kind and on its executor.

    The divisor of a mean is the sum of these, and every array of one kernel
    call has to be where the values are.
    """
    if isinstance(values, np.ndarray):
        return np.ones(len(values))
    import neon  # noqa: PLC0415  # NeoN is only needed on the NeoN path

    return neon.ScalarVector(values.exec(), values.size(), 1.0)
