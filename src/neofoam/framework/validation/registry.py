# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Honest case validation — a registry of isolated checks over a case directory.

Each check is a small unit that reads one aspect of a case and yields
:class:`Finding` objects. :meth:`CheckRegistry.run` reports
``ok = (no error findings) AND (every check ran)``: a check that cannot run — an
``Unreadable`` leaf from :mod:`neofoam.io.dictread`, or a check that raises — becomes
an *error* finding, never a silent skip. This is the structural completion of the
silent-false-success fix in ``validate_case``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel

__all__ = [
    "Finding",
    "ValidationReport",
    "CaseContext",
    "Check",
    "CheckRegistry",
]


class Finding(BaseModel):
    """One issue found by a check — with a concrete fix where possible."""

    level: Literal["error", "warning"]
    file: str
    message: str
    fix: str | None = None


class ValidationReport(BaseModel):
    """The result of a static pre-flight check of a case (no OpenFOAM run)."""

    ok: bool
    findings: list[Finding]


@dataclass(frozen=True)
class CaseContext:
    """What a check reads: the case directory + the solver spec it targets."""

    case: Path
    solver: Any


Check = Callable[[CaseContext], list[Finding]]


@dataclass
class CheckRegistry:
    """An ordered set of named, isolated checks over a :class:`CaseContext`."""

    checks: list[tuple[str, Check]] = field(default_factory=list)

    def add(self, name: str, check: Check) -> None:
        """Register ``check`` under ``name`` (order preserved for stable reports)."""
        self.checks.append((name, check))

    def run(self, ctx: CaseContext) -> ValidationReport:
        """Run every check; ``ok`` iff no error finding **and** no check failed to run.

        A check that raises is not allowed to abort the run or silently pass — it is
        turned into an error finding, so ``ok`` becomes ``False``. Combined with each
        check's own ``Unreadable ⇒ error`` escalation, this makes "every check ran"
        structural rather than assumed.
        """
        findings: list[Finding] = []
        for name, check in self.checks:
            try:
                findings.extend(check(ctx))
            except Exception as exc:  # a check that cannot run is an error, not a skip
                findings.append(
                    Finding(
                        level="error",
                        file=f"<check:{name}>",
                        message=f"validation check {name!r} could not run: {exc}",
                    )
                )
        ok = not any(f.level == "error" for f in findings)
        return ValidationReport(ok=ok, findings=findings)
