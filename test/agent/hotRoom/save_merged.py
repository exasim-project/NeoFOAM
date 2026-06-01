# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""Merge configs that target the same file and write each file once.

Multiple ``BaseConfig`` classes can declare the same ``io_config.file``:
``TransportPropertiesConfig`` and ``BoussinesqConfig`` both land in
``constant/transportProperties``; ``Pimple_fvSchemes`` and
``boussinesq_fvSchemes`` both land in ``system/fvSchemes``. Calling
``instance.save()`` in sequence loses the earlier writer because
``OpenFOAMStrategy.write`` does ``root_dict.clear()`` before emitting —
so the file on disk reflects only the last contributor.

:func:`save_merged` groups instances by their declared target file, deep-merges
each instance's ``model_dump(by_alias=True, exclude_none=True)`` payload, and
hands the merged dict to the strategy's writer (reusing
``OpenFOAMStrategy._write``'s dispatch). Collisions on the same key are
resolved last-wins in iteration order, so callers control precedence by
ordering the instances they pass in.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Union

import pybFoam as pyf
from pydantic import BaseModel

from neofoam.io.strategies.openfoam_strategy import OpenFOAMStrategy


def _deep_merge(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    """Recursive last-wins merge: keys in ``src`` overwrite ``dst`` in place."""
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_merge(dst[key], value)
        else:
            dst[key] = value
    return dst


def save_merged(
    instances: Iterable[BaseModel],
    case_dir: Union[Path, str],
) -> dict[str, list[str]]:
    """Group ``instances`` by ``io_config.file`` and write one merged file per group.

    Each instance contributes its ``model_dump(by_alias=True, exclude_none=True)``
    payload. Within a group, keys are last-wins in iteration order.

    Args:
        instances: Configs to persist. Instances without ``io_config`` (no IO
            strategy registered) are skipped silently.
        case_dir: Case directory; each ``io_config.file`` resolves relative to it.

    Returns:
        Mapping of ``relative_file_path`` → contributing class names (for logging).

    Raises:
        TypeError: If two contributors to the same file declare different
            strategy classes (e.g. one YAML + one OpenFOAM into the same path).
        NotImplementedError: For strategies without a merged-write path
            (only :class:`OpenFOAMStrategy` is wired up so far — extend
            :func:`_write_merged_payload` when adding more).
    """
    groups: dict[str, list[BaseModel]] = defaultdict(list)
    for inst in instances:
        io = getattr(type(inst), "io_config", None)
        if io is None:
            continue
        groups[io.file].append(inst)

    case_dir_path = Path(case_dir)
    report: dict[str, list[str]] = {}
    for file, contribs in groups.items():
        merged: dict[str, Any] = {}
        for inst in contribs:
            _deep_merge(
                merged,
                inst.model_dump(mode="python", exclude_none=True, by_alias=True),
            )

        writers = {type(type(c).io_config.writer) for c in contribs}
        if len(writers) > 1:
            names = ", ".join(sorted(s.__name__ for s in writers))
            raise TypeError(
                f"Cannot merge into {file!r}: contributors use mixed strategies ({names})"
            )

        strategy = type(contribs[0]).io_config.writer
        _write_merged_payload(strategy, case_dir_path / file, merged)
        report[file] = [type(c).__name__ for c in contribs]
    return report


def _write_merged_payload(strategy: Any, path: Path, data: dict[str, Any]) -> None:
    """Strategy-aware emit for a pre-merged dict.

    Mirrors the file/dir handling in each strategy's own ``write`` but skips the
    per-instance ``model_dump`` step (the caller already merged in Python).
    """
    if isinstance(strategy, OpenFOAMStrategy):
        path.parent.mkdir(parents=True, exist_ok=True)
        root = pyf.dictionary.read(str(path)) if path.exists() else pyf.dictionary()
        root.clear()
        # ``_write`` is the strategy's own dict→pybFoam dispatch (handles bool
        # → yes/no, nested sub-dicts, unsupported-type errors); reusing it
        # keeps merged writes identical in encoding to single-instance writes.
        strategy._write(root, data)
        root.write(str(path))
        return
    raise NotImplementedError(
        f"save_merged: no merged-write path for {type(strategy).__name__}"
    )
