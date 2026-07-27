# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""dropin/ depends on neofoam through exactly one seam, and this pins it.

plans/workflow-composable-and-benchmarkable.md §1: the study's entire coupling
to neofoam is casebuild + io.DictFile (+ `neofoam solver <backend>` as a
subprocess from a generated Allrun, never imported). That triple is the public,
stable contract a future repo split rests on -- if a study ever needs a fourth
thing, that is the signal to either promote it deliberately or keep it local,
not to add it here quietly. This test is the guard: it fails the moment
dropin/ reaches for anything else under neofoam.

Path-based, no OpenFOAM, no snakemake -- runs everywhere.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

_DROPIN = Path(__file__).resolve().parents[1] / "dropin"

#: Every `neofoam.X` name a dropin module may import. `None` means "any name
#: from this module is allowed" (casebuild's whole __all__ is fair game).
_ALLOWED: dict[str, set[str] | None] = {
    "neofoam.tooling.casebuild": None,
    "neofoam.io": {"DictFile"},
}


def _dropin_modules() -> list[Path]:
    return sorted(_DROPIN.rglob("*.py"))


def _neofoam_imports(path: Path) -> list[tuple[str, str]]:
    """(module, name) for every neofoam import in *path*, however it is spelled."""
    tree = ast.parse(path.read_text(), filename=str(path))
    found: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("neofoam"):
            module = node.module or ""
            for alias in node.names:
                found.append((module, alias.name))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("neofoam"):
                    found.append((alias.name, "*"))
    return found


def test_dropin_has_python_files_to_check() -> None:
    """Guard the guard: an empty module list would make the check below vacuous."""
    assert _dropin_modules()


def test_dropin_imports_only_the_pinned_neofoam_api() -> None:
    violations = []
    for path in _dropin_modules():
        for module, name in _neofoam_imports(path):
            if module not in _ALLOWED:
                violations.append(f"{path.relative_to(_DROPIN)}: {module}.{name}")
                continue
            allowed_names = _ALLOWED[module]
            if allowed_names is not None and name not in allowed_names:
                violations.append(f"{path.relative_to(_DROPIN)}: {module}.{name}")
    assert not violations, (
        "dropin/ must depend on neofoam only through casebuild + io.DictFile "
        "(+ the neofoam CLI as a subprocess, checked separately below):\n" + "\n".join(violations)
    )


def test_the_neofoam_cli_is_the_third_seam_and_it_actually_exists() -> None:
    """`neofoam solver <backend>` is the contract swap_solver() writes into Allrun.

    Not exercised by import (nothing in dropin/ imports the CLI), so it is pinned
    here as a subprocess smoke test instead -- --help needs no sourced OpenFOAM,
    no run, just confirms the command surface the generated Allrun depends on is
    real. Mirrors config.yaml's `apps:` entries (see the two studies' config.yaml).
    """
    result = subprocess.run(
        [sys.executable, "-m", "neofoam.cli.app", "solver", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "incompressiblefluid" in result.stdout
    assert "incompressiblevof" in result.stdout
