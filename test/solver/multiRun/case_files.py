# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The multiRun test cases, copied to a run directory."""

import shutil
from pathlib import Path
from typing import Optional

CASES = Path(__file__).parent / "cases"


def copy_case(name: str, run_dir: Path, overlay: Optional[str] = None) -> Path:
    """``cases/<name>`` copied to ``run_dir``, the files of ``cases/<overlay>`` laid over it."""
    shutil.copytree(CASES / name, run_dir)
    if overlay is not None:
        shutil.copytree(CASES / overlay, run_dir, dirs_exist_ok=True)
    return run_dir
