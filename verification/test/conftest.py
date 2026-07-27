# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""verification/dropin/ has no package __init__.py above it (a deliberate
namespace package — see verification/README.md), so `from verification.dropin
import ...` needs the repo root on sys.path. Bare `pytest` (this repo's
convention — doc/reference/build-and-test.rst) does not add it for a fresh
top-level namespace package; verified with a throwaway repro before writing
this. Snakemake gets the same value a different way (rules/header.smk computes
it from CONFIG and PYTHONPATH for its subprocess rules) — this is the pytest
side of the same problem.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
