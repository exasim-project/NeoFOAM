# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The built-in case-build steps: ``patch``, ``configs`` (meshing lives in ``meshing.py``).

Each is a free function returning a :data:`~neofoam.tooling.casebuild.pipeline.Step`
(``Callable[[CaseDir], None]``) so it can be injected into a pipeline with ``|``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Optional, Union

from pydantic import BaseModel

from neofoam.tooling.casebuild.pipeline import CaseDir, Step
from neofoam.io import DictFile, write_configs


def _addr(key: str) -> Union[str, tuple[str, ...]]:
    """A dotted key becomes a :class:`DictFile` tuple address; a plain key stays."""
    return tuple(key.split(".")) if "." in key else key


def patch(
    rel_path: str,
    overrides: Optional[Mapping[str, object]] = None,
    /,
    *,
    remove: Iterable[str] = (),
    **kwargs: object,
) -> Step:
    """Set and/or remove entries in the dict at *rel_path*.

    Merges the *overrides* mapping with keyword arguments (kwargs win on conflict)
    to form the keys to set; the keyword-only ``remove`` lists keys to delete
    (its inverse -- forking one base in the "key absent" direction without
    committing a second template, e.g.
    ``patch("system/controlDict", remove=["adjustTimeStep"])``). Both accept
    dotted keys addressing sub-dicts, e.g.
    ``patch("system/fvSolution", **{"PIMPLE.nCorrectors": 2})``. Removing an
    absent key is a no-op. The file's format (OpenFOAM / JSON / YAML) is chosen
    by suffix -- see :class:`neofoam.io.DictFile`.
    """
    merged: dict[str, object] = {**(overrides or {}), **kwargs}
    removals = list(remove)

    def step(case: CaseDir) -> None:
        d = DictFile(case.path / rel_path)
        for key, value in merged.items():
            d.set(_addr(key), value)
        for key in removals:
            d.remove(_addr(key))
        d.write()

    return step


def configs(*instances: BaseModel) -> Step:
    """Write pydantic case configs to disk via :func:`neofoam.io.write_configs`."""

    def step(case: CaseDir) -> None:
        write_configs(instances, case_dir=case.path)

    return step
