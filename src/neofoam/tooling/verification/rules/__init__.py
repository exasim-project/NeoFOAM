# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The packaged Snakemake rules for the tutorial verification suite.

Same authoring convention as :mod:`neofoam.tooling.workflow.rules`: each ``.smk``
file holds one rule whose ``shell:`` body is a single ``python -m
neofoam.tooling.verification.runner`` call, and the study ``Snakefile``
``include:``\\s them by path via :func:`rules_dir`. They are *not* the mesh
pipeline's rules — a drop-in tutorial runs its own ``Allrun``, so there is no
mesh chain here — hence a separate directory rather than an entry in that
pipeline's ``default_registry``.

The header globals every rule consumes (``CONFIG``, ``WORK``, ``RESULTS``,
``REPORT``, ``IDS``, ``THREADS``) are defined by the study Snakefile.
"""

from __future__ import annotations

import importlib.resources
from pathlib import Path

__all__ = ["rules_dir"]


def rules_dir() -> Path:
    """The installed directory holding the packaged verification ``.smk`` files."""
    return Path(str(importlib.resources.files("neofoam.tooling.verification.rules")))
