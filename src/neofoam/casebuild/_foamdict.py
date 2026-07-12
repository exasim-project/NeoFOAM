# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Value-level patching of OpenFOAM dictionary files through the pybFoam parser.

:func:`apply_overrides` reads a dict file, sets each ``key -> value`` (dotted keys
address sub-dicts, e.g. ``"PIMPLE.nCorrectors"``), and writes it back. Varying a
case this way goes through OpenFOAM's own reader/writer — not text patching — and
reaches keys no pydantic config models. The value dispatch mirrors
:data:`neofoam.io.strategies.openfoam_strategy.WRITE_DISPATCH` (``bool`` becomes the
OpenFOAM Switch ``yes``/``no``; the rest forward to ``set`` for pybFoam to stringify).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pybFoam as pyf


# Runtime-type → setter, keyed on the *exact* type so ``bool`` (a subclass of
# ``int``) picks the Switch encoding rather than the int path.
_SET_DISPATCH: dict[type, Callable[[Any, str, Any], None]] = {
    str: lambda d, key, v: d.set(key, v),
    int: lambda d, key, v: d.set(key, v),
    float: lambda d, key, v: d.set(key, v),
    bool: lambda d, key, v: d.set(key, "yes" if v else "no"),
}


def _set_value(target: Any, key: str, value: object) -> None:
    writer = _SET_DISPATCH.get(type(value))
    if writer is None:
        raise TypeError(
            f"Unsupported override value type for key '{key}': "
            f"{type(value).__name__} (supported: str, int, float, bool)"
        )
    writer(target, key, value)


def apply_overrides(path: Path, overrides: Mapping[str, object]) -> None:
    """Set each override in the OpenFOAM dict at *path*, in place.

    Dotted keys walk (creating if absent) sub-dicts. Raises ``FileNotFoundError``
    if *path* is not an existing file — patching presupposes a dict to edit.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Cannot patch missing dictionary file: {path}")

    root = pyf.dictionary.read(str(path))
    for dotted_key, value in overrides.items():
        *parents, leaf = dotted_key.split(".")
        target = root
        for part in parents:
            # subDictOrAdd returns a detached handle; re-fetch via subDict so
            # mutations propagate to the parent (see openfoam_strategy._write).
            target.subDictOrAdd(part)
            target = target.subDict(part)
        _set_value(target, leaf, value)
    root.write(str(path))
