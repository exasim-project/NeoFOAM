# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Typed OpenFOAM-dict *leaf* reader — the honest read layer under validation.

Reads one dict section (or one entry) via ``pybFoam.dictionary`` and classifies
**each leaf** as :class:`Value` or :class:`Unreadable`. A leaf that cannot be
rendered as text becomes an explicit :class:`Unreadable` rather than a swallowed
skip that truncates the whole section — the mechanism behind the silent
false-success in ``validate_case``. Frontend-agnostic: ``import pybFoam`` is lazy
inside the functions, so ``import neofoam.io`` pulls no ``pybFoam``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Union

__all__ = [
    "Value",
    "Unreadable",
    "Leaf",
    "read_section",
    "read_entry",
    "read_keys",
    "read_toplevel",
]


@dataclass(frozen=True)
class Value:
    """A dict leaf successfully rendered as OpenFOAM text (a single token string)."""

    text: str


@dataclass(frozen=True)
class Unreadable:
    """A present leaf/section that could not be rendered as text.

    Carries *why* so a caller records an explicit finding instead of silently
    skipping it — the opposite of the old ``except Exception: pass``.
    """

    reason: str


Leaf = Union[Value, Unreadable]


def _leaf(render: Callable[[], str]) -> Leaf:
    """Classify one leaf read: its text, or an explicit :class:`Unreadable`.

    The read primitive is passed as a thunk so its failure is isolated to a single
    leaf — one bad leaf never truncates its section.
    """
    try:
        return Value(text=str(render()))
    except Exception as exc:  # any backend read failure is made explicit, not swallowed
        return Unreadable(reason=str(exc).strip() or type(exc).__name__)


def read_section(path: Path, section: str) -> dict[str, dict[str, Leaf]]:
    """Read ``section`` into ``{sub-name -> {leaf-key -> Value | Unreadable}}``.

    A missing file or a missing ``section`` yields ``{}`` (a genuine absence other
    checks own). A present sub-entry whose leaf cannot be rendered keeps every
    sibling leaf and every other sub-entry — the leaf alone becomes
    :class:`Unreadable`.
    """
    out: dict[str, dict[str, Leaf]] = {}
    if not path.is_file():
        return out
    import pybFoam as pyf

    # Intentionally NO try/except around the parse here (unlike read_keys/
    # read_toplevel): on this backend a genuinely corrupt FOAM dict aborts the
    # process uncatchably, so guarding it would be dead code — the real seam is
    # leaf-level (_leaf turns an unrenderable value into Unreadable). Do not
    # "helpfully" add a guard: it cannot catch a real abort.
    root: Any = pyf.dictionary.read(str(path))  # pybFoam dictionary: dynamic access
    if not root.found(section):
        return out
    block = root.subDict(section)
    for key in block.toc():
        name = str(key)
        if not block.isDict(name):
            continue
        sub = block.subDict(name)
        leaves: dict[str, Leaf] = {}
        for k in sub.toc():
            leaf_key = str(k)
            if sub.isDict(leaf_key):
                continue
            # _leaf invokes the thunk eagerly right here, before ``leaf_key``
            # advances, so no late-binding capture guard is needed.
            leaves[leaf_key] = _leaf(lambda: sub.get[str](leaf_key))
        out[name] = leaves
    return out


def read_entry(path: Path, section: str, key: str) -> Optional[Leaf]:
    """One leaf of ``section`` (e.g. ``divSchemes``/``div(phi,U)``).

    ``None`` when the file, ``section``, or ``key`` is absent; :class:`Unreadable`
    when the value is present but not renderable (absence and unreadability are
    different answers).
    """
    if not path.is_file():
        return None
    import pybFoam as pyf

    root: Any = pyf.dictionary.read(str(path))  # pybFoam dictionary: dynamic access
    if not root.found(section):
        return None
    block = root.subDict(section)
    for key_tok in block.toc():
        if str(key_tok) == key:
            return _leaf(lambda: block.get[str](key))
    return None


def read_keys(path: Path) -> Union[frozenset[str], Unreadable, None]:
    """Top-level key names of a dict file (its ``toc``) — presence without values.

    ``None`` when the file is absent (a genuine absence other checks own);
    :class:`Unreadable` when the file is present but will not parse, so a caller that
    needs mere *presence* of a key (not its value) still distinguishes a corrupt dict
    from a key that is simply not there. No leaf value is rendered, so a dimensioned
    entry (``beta``/``TRef``) is never misclassified as unreadable.
    """
    if not path.is_file():
        return None
    import pybFoam as pyf

    try:
        root: Any = pyf.dictionary.read(str(path))  # pybFoam dictionary: dynamic access
    except Exception as exc:  # a dict that will not parse is explicit, not swallowed
        return Unreadable(reason=str(exc).strip() or type(exc).__name__)
    return frozenset(str(k) for k in root.toc())


def read_toplevel(path: Path, key: str) -> Optional[Leaf]:
    """One *top-level* (non-sectioned) entry as a :class:`Leaf` — analogue of
    :func:`read_entry` for keys that live at the dict root.

    ``None`` when the file or ``key`` is absent; :class:`Unreadable` when the file
    will not parse **or** the entry is present but not renderable as text.
    """
    if not path.is_file():
        return None
    import pybFoam as pyf

    try:
        root: Any = pyf.dictionary.read(str(path))  # pybFoam dictionary: dynamic access
    except Exception as exc:
        return Unreadable(reason=str(exc).strip() or type(exc).__name__)
    if not root.found(key):
        return None
    return _leaf(lambda: root.get[str](key))
