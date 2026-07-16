# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Write a batch of configs, combining co-owners of a file into one write.

Several ``BaseConfig`` classes can declare the same ``io_config.file``:
``TransportPropertiesConfig`` + ``BoussinesqConfig`` → ``constant/transportProperties``;
the PIMPLE + boussinesq ``fvSchemes`` slices → ``system/fvSchemes``. The OpenFOAM
writer clears a file before rewriting it (so re-saving a single config drops keys
it no longer carries — e.g. a BC flipped from ``fixedValue`` to ``zeroGradient``
must not leave a stale ``value``). That same clear would make sequential
per-config writes clobber each other, so :func:`write_configs` first groups
instances by their target file and deep-merges each group's
``model_dump(by_alias=True, exclude_none=True)`` into one payload, then writes
that payload once. Within a file, keys are last-wins in iteration order, so
callers control precedence by ordering the instances they pass in.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Union

from pydantic import BaseModel

from neofoam.io.dictfile import _write_payload
from neofoam.io.strategies.openfoam_strategy import OpenFOAMStrategy
from neofoam.io.strategies.yaml_strategy import YAMLStrategy

__all__ = ["write_configs"]


def _deep_merge(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    """Recursive last-wins merge: keys in ``src`` overwrite ``dst`` in place."""
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_merge(dst[key], value)
        else:
            dst[key] = value
    return dst


def write_configs(
    instances: Iterable[BaseModel],
    case_dir: Union[Path, str],
) -> dict[str, list[str]]:
    """Group ``instances`` by ``io_config.file`` and write one file per group.

    Each instance contributes its ``model_dump(by_alias=True, exclude_none=True)``
    payload; co-owners of a file are deep-merged (last-wins per key).

    Args:
        instances: Configs to persist. Instances without ``io_config`` (no IO
            strategy registered) are skipped.
        case_dir: Case directory; each ``io_config.file`` resolves relative to it.

    Returns:
        Mapping of ``relative_file_path`` → contributing class names.

    Raises:
        TypeError: If two contributors to the same file declare different
            strategy classes (e.g. one YAML + one OpenFOAM into the same path).
        NotImplementedError: For strategies without a merged-write path (only
            :class:`OpenFOAMStrategy` is wired up; extend
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
                inst.model_dump(
                    mode="python",
                    exclude_none=True,
                    by_alias=True,
                    # OpenFOAM literal mapping for FieldValue types (write_configs
                    # only writes OpenFOAM strategies — see _write_merged_payload).
                    context={"format": "openfoam"},
                ),
            )

        writers = {
            type(type(c).io_config.writer)  # type: ignore[attr-defined]
            for c in contribs
        }
        if len(writers) > 1:
            names = ", ".join(sorted(s.__name__ for s in writers))
            raise TypeError(
                f"Cannot write {file!r}: contributors use mixed strategies ({names})"
            )

        strategy = type(contribs[0]).io_config.writer  # type: ignore[attr-defined]
        _write_merged_payload(strategy, case_dir_path / file, merged)
        report[file] = [type(c).__name__ for c in contribs]
    return report


def _write_merged_payload(strategy: Any, path: Path, data: dict[str, Any]) -> None:
    """Emit a pre-merged (whole-file) payload through the shared write engine.

    The caller already merged co-owners in Python, so this hands the payload to
    :func:`neofoam.io.dictfile._write_payload` -- the same engine behind
    ``BaseConfig.save`` -- which injects a ``FoamFile`` header for header-less
    dict configs. A payload that round-tripped through ``load(...)`` (e.g. the
    AI-fill push) carries ``FoamFile`` as its last key; hoist it to the front so
    the header leads the file (OpenFOAM rejects a non-leading header).
    """
    if isinstance(strategy, OpenFOAMStrategy):
        if "FoamFile" in data:
            data = {"FoamFile": data["FoamFile"], **data}
        _write_payload(path, data, (), "openfoam")
        return
    if isinstance(strategy, YAMLStrategy):
        # Whole-file YAML (e.g. ``system/preprocess.yaml``): no ``FoamFile`` header,
        # dumped through the same engine the OpenFOAM path uses (it dispatches on the
        # ``.yaml`` suffix / ``fmt``). Grouping is per file, so this is a whole-file
        # write like the OpenFOAM branch.
        _write_payload(path, data, (), "yaml")
        return
    raise NotImplementedError(
        f"write_configs: no merged-write path for {type(strategy).__name__}"
    )
