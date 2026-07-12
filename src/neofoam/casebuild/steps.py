# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The built-in case-build steps: ``patch``, ``configs`` (meshing lives in ``meshing.py``).

Each is a free function returning a :data:`~neofoam.casebuild.pipeline.Step`
(``Callable[[CaseDir], None]``) so it can be injected into a pipeline with ``|``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Optional

from pydantic import BaseModel

from neofoam.casebuild._foamdict import apply_overrides
from neofoam.casebuild.pipeline import CaseDir, Step
from neofoam.io import write_configs


def patch(
    rel_path: str,
    overrides: Optional[Mapping[str, object]] = None,
    /,
    **kwargs: object,
) -> Step:
    """Override entries in the OpenFOAM dict at *rel_path*.

    Merges the *overrides* mapping with keyword arguments (kwargs win on conflict).
    Dotted keys address sub-dicts, e.g. ``patch("system/fvSolution", **{"PIMPLE.nCorrectors": 2})``.
    """
    merged: dict[str, object] = {**(overrides or {}), **kwargs}

    def step(case: CaseDir) -> None:
        apply_overrides(case.path / rel_path, merged)

    return step


def configs(*instances: BaseModel) -> Step:
    """Write pydantic case configs to disk via :func:`neofoam.io.write_configs`."""

    def step(case: CaseDir) -> None:
        write_configs(instances, case_dir=case.path)

    return step
