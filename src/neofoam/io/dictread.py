# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Typed OpenFOAM-dict *leaf* reader — the honest read layer under validation.

Reads one dict section (or one entry) via ``pybFoam.dictionary`` and classifies
**each leaf** as :class:`Value` or :class:`Unreadable`. A leaf that cannot be
rendered as text becomes an explicit :class:`Unreadable` rather than a swallowed
skip that truncates the whole section — the mechanism behind the silent
false-success in ``validate_case``.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Union

import pybFoam as pyf

__all__ = [
    "Value",
    "Unreadable",
    "Leaf",
    "foam_case",
    "set_foam_case",
    "read_section",
    "read_entry",
    "read_keys",
    "read_toplevel",
]

#: What ``argList`` sets when a solver starts, and OpenFOAM's path tags expand from.
_FOAM_CASE = "FOAM_CASE"


@contextmanager
def foam_case(case: Union[Path, str]) -> Iterator[None]:
    """Expose ``case`` as ``$FOAM_CASE`` while its dictionaries are read.

    OpenFOAM's path tags (``<system>``, ``<constant>``, ``<case>``) expand from
    ``$FOAM_CASE`` — set by ``argList`` when a solver starts, and by nothing at all in
    a plain Python process. Unset, the ``#include "<system>/meshQualityDict"`` that
    OpenFOAM's own ``snappyHexMeshDict.cfg`` carries degrades to a bare relative path,
    which resolves against the *including* file (somewhere under ``etc/``) instead of
    the case — so a case including it cannot be read at all, and the failure is a
    process abort rather than an exception::

        with foam_case(case_dir):
            configs = load_case_from_disk(case_dir, solver=solver)

    The variable is process-global, so whatever was there is restored on the way out.
    """
    previous = os.environ.get(_FOAM_CASE)
    set_foam_case(case)
    try:
        yield
    finally:
        set_foam_case(previous)


def set_foam_case(case: Union[Path, str, None]) -> None:
    """Point ``$FOAM_CASE`` at ``case`` until something else moves it; clear it on ``None``.

    The lasting counterpart of :func:`foam_case`, for an application that *has* a
    current case (the wizard, once one is loaded) rather than one reading a case it
    was handed. ``argList`` does this for a solver; nothing does it for a plain
    Python process, so an app that reads case dictionaries sets it itself.
    """
    if not case:
        os.environ.pop(_FOAM_CASE, None)
        return
    os.environ[_FOAM_CASE] = str(Path(case).resolve())


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
    :class:`Unreadable`. A nested block is skipped, except OpenFOAM's
    name-or-dictionary entry, which reads as the name it holds.
    """
    out: dict[str, dict[str, Leaf]] = {}
    if not path.is_file():
        return out

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
            # OpenFOAM's name-or-dictionary entry (``preconditioner { preconditioner
            # GAMG; … }``) reads as its inner name; any other nested block is skipped.
            holder = sub.subDict(leaf_key) if sub.isDict(leaf_key) else sub
            if not holder.found(leaf_key):
                continue
            # _leaf invokes the thunk eagerly right here, before ``leaf_key``
            # advances, so no late-binding capture guard is needed.
            leaves[leaf_key] = _leaf(lambda: holder.get[str](leaf_key))
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

    try:
        root: Any = pyf.dictionary.read(str(path))  # pybFoam dictionary: dynamic access
    except Exception as exc:
        return Unreadable(reason=str(exc).strip() or type(exc).__name__)
    if not root.found(key):
        return None
    return _leaf(lambda: root.get[str](key))
