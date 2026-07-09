# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Automated guard: requirement/task IDs must not leak into the VoF test source.

Requirement, task, and review-finding identifiers belong in the plan's traceability
table, not in test code (names, comments, or docstrings). A prior iteration leaked
``Task 6a``/``Task 6b`` into comments; this test fails the suite if any such ID
re-appears in any ``test_*.py`` in this directory, replacing the manual grep that could
silently drift.
"""

from __future__ import annotations

import re
from pathlib import Path

_TEST_DIR = Path(__file__).parent
_GUARD_FILE = Path(__file__).name

# Requirement/task/finding/slice ID shapes. The prefix set is chosen to not
# false-positive on legitimate tokens in the guarded source (e.g. Vec3, U, hex commit
# fragments, 2D/3D). Review-finding IDs are matched as F followed by ONE or TWO digits
# (F1..F99) so they do not collide with three-digit ruff codes (e.g. ``noqa: F401``).
_ID_PATTERN = re.compile(
    r"\b(?:Task|FR|IF|TC|R)[ _-]?\d+[a-z]?\b"  # requirement / task IDs
    r"|\bF[ _-]?\d{1,2}\b"  # review-finding IDs F1..F99 (not ruff FNNN)
    r"|\b[A-D]\d+\b"  # bare slice IDs A1..D9
)


def _guarded_files() -> list[Path]:
    files = sorted(p for p in _TEST_DIR.glob("test_*.py") if p.name != _GUARD_FILE)
    assert files, "guard found no test_*.py files to scan — wrongly scoped"
    return files


def test_no_requirement_ids_in_test_source() -> None:
    hits: list[str] = []
    for f in _guarded_files():
        for lineno, line in enumerate(f.read_text().splitlines(), start=1):
            if _ID_PATTERN.search(line):
                hits.append(f"{f.name}:{lineno}: {line.strip()}")
    assert not hits, "requirement IDs leaked into test source:\n" + "\n".join(hits)


def test_id_pattern_fires_on_a_planted_id() -> None:
    # Negative self-test for the regex: it must match planted requirement IDs, so a green
    # run of the guard means "clean", not "pattern never matches anything".
    assert _ID_PATTERN.search("# Task 6a: validation")
    assert _ID_PATTERN.search("see FR3 for details")
    assert _ID_PATTERN.search("# close F1 health-gate")


def test_id_guard_scan_wiring_fires(tmp_path: Path) -> None:
    # Negative self-test for the SCAN (not just the regex): a planted ID in a scanned file
    # must be flagged, so a wrongly scoped _guarded_files() can't false-green.
    planted = tmp_path / "test_planted.py"
    planted.write_text("# resolves finding F3 and Task 6\n")
    hits = [ln for ln in planted.read_text().splitlines() if _ID_PATTERN.search(ln)]
    assert hits, "scan wiring failed to flag a planted ID"
