# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Temporary: verification/dropin/ has no package __init__.py above it, so a
bare `from verification.dropin... import ...` needs the repo root on sys.path
(bare `pytest` does not add it for a fresh namespace package — verified). This
file is superseded by verification/test/conftest.py once these tests move there
in the next commit, and is deleted along with the rest of this directory then.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
