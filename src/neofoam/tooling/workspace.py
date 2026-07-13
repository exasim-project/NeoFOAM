# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The trust boundary: confine an untrusted case_id to a workspace root.

A :class:`Workspace` resolves an untrusted (relative) ``case_id`` under a fixed root
directory and rejects any path that escapes it — an absolute path, a ``..`` traversal
that leaves the root, or a symlink whose target resolves outside — raising a typed
:class:`CaseAccessError`. Frontends construct a ``Workspace`` at their edge (where
external input enters) and resolve *before* touching the case, so path confinement
lives at the boundary, not deep in the domain.

Stdlib-only on purpose: importing this module pulls no trame / fastmcp / pybFoam, so
it is the first, dependency-light module of the ``tooling`` frontend layer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


class CaseAccessError(ValueError):
    """An untrusted case path escaped its workspace root, or is absent/unreadable.

    Subclasses :class:`ValueError` so existing ``ValueError`` handling (and the MCP
    ``ValueError -> tool error`` surfacing) keeps working while callers that care can
    catch the precise trust-boundary type.
    """


@dataclass(frozen=True)
class Workspace:
    """Confines an untrusted ``case_id`` under :attr:`root`."""

    root: Path

    @classmethod
    def at(cls, root: str | Path) -> "Workspace":
        """Build a workspace rooted at ``root`` (normalized to a resolved abs path)."""
        return cls(root=Path(root).resolve())

    def resolve(self, case_id: str | Path, *, kind: str = "case_id") -> Path:
        """Resolve an untrusted (relative) ``case_id`` under :attr:`root`.

        Rejects an absolute ``case_id``, an empty/``.`` id that would name the root
        itself, and any ``..``/symlink path that resolves outside the root — all with
        :class:`CaseAccessError`. The returned path need **not** exist (this is the
        confinement seam for a write *target*). ``kind`` labels the value in error
        messages (e.g. ``"source_dir"``) so a caller can tell which path was rejected.
        """
        candidate = Path(case_id)
        if candidate.is_absolute():
            raise CaseAccessError(
                f"{kind} must be a relative path, got absolute: {case_id!r}"
            )
        if not candidate.parts:  # "" and "." both normalize to the root — never a case
            raise CaseAccessError(
                f"{kind} must name a case under the root, got empty/'.': {case_id!r}"
            )
        resolved = (self.root / candidate).resolve()
        if not self._within(resolved):
            raise CaseAccessError(
                f"{kind} escapes the workspace root {self.root}: {case_id!r}"
            )
        return resolved

    def resolve_existing(
        self, case_id: str | Path, *, require_dir: bool = True, kind: str = "case_id"
    ) -> Path:
        """:meth:`resolve` + require the target exists, is a dir, and is readable.

        The pre-flight for a *source* case: a missing/unreadable source raises
        :class:`CaseAccessError` before any downstream work (e.g. an LLM call).
        ``kind`` labels the value in error messages (e.g. ``"source_dir"``).
        """
        resolved = self.resolve(case_id, kind=kind)
        if not resolved.exists():
            raise CaseAccessError(f"{kind} does not exist: {case_id!r}")
        if require_dir and not resolved.is_dir():
            raise CaseAccessError(f"{kind} is not a directory: {case_id!r}")
        if not os.access(resolved, os.R_OK):
            raise CaseAccessError(f"{kind} is not readable: {case_id!r}")
        return resolved

    def _within(self, resolved: Path) -> bool:
        """True when ``resolved`` is the root or lies beneath it."""
        try:
            resolved.relative_to(self.root.resolve())
            return True
        except ValueError:
            return False
