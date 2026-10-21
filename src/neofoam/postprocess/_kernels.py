# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The post-processing kernels, resolved on first use rather than at import.

``neofoam.neofoam_bindings`` is ``None`` whenever the compiled stack is not
built — a plain source checkout, or ``PYTHONPATH=src`` in front of a
non-editable install, which is how the parallel drivers run. Dereferencing it at
import time would make ``import neofoam`` itself fail there (``neofoam.io``
reaches this package), so the kernels are looked up per attribute, the way every
other ``nfb`` caller in the package does it, and a missing stack is reported as a
named :class:`ImportError` instead of an ``AttributeError`` on ``None``::

    from neofoam.postprocess import _kernels as pp

    pp.sum(values, n_groups)
"""

from __future__ import annotations

from typing import Any

import neofoam


def __getattr__(name: str) -> Any:
    """One kernel of ``neofoam_bindings.postprocess``, or why it is unavailable."""
    bindings = neofoam.neofoam_bindings
    if bindings is None:
        raise ImportError(
            f"postProcess needs the compiled NeoFOAM bindings for {name!r}, but "
            "neofoam.neofoam_bindings is not built in this interpreter — install "
            "the package (pip install .) instead of importing it from src/"
        )
    return getattr(bindings.postprocess, name)
