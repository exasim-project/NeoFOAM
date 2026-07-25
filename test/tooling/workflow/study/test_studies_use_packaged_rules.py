# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Every study under ``verification/foam_tutorials`` is three files on one pipeline.

The property this guards is the point of the whole packaged rule set: a study
directory carries *what to sweep* (config.yaml + discover.py) and nothing about
*how*. The failure it catches is drift — a study copying the rules back onto disk,
or a new study cloned from an old one still importing the deleted
``neofoam.tooling.verification``. Both look fine until someone edits the shared
pipeline and one study silently keeps the old behaviour.

Path-based on purpose: no OpenFOAM, no snakemake, so it runs everywhere.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_STUDIES_ROOT = Path(__file__).parents[4] / "verification" / "foam_tutorials"

#: The only files a study directory may track. Run artifacts (cases/, results/,
#: work/, report.html) are gitignored and not checked here.
_ALLOWED = {"Snakefile", "config.yaml", "discover.py"}


def _studies() -> list[Path]:
    return sorted(p for p in _STUDIES_ROOT.iterdir() if (p / "Snakefile").is_file())


def test_the_studies_are_discoverable() -> None:
    """Guard the guard: an empty study list would make every check below vacuous."""
    assert [p.name for p in _studies()] == ["incompressibleFluid", "incompressibleVoF"]


def test_the_old_verification_package_is_gone() -> None:
    """Its engine is workflow.study and its rules are workflow/rules/study*.smk.

    Checked on disk rather than by importing: an editable install keeps a redirect
    map built at install time, so a deleted package can still resolve to a stale
    path until the next ``pip install -e``. The source tree is the truth.
    """
    tooling = Path(__import__("neofoam.tooling", fromlist=["__file__"]).__file__).parent
    assert not (tooling / "verification").exists()


@pytest.mark.parametrize("study", _studies(), ids=lambda p: p.name)
def test_a_study_includes_the_packaged_pipeline(study: Path) -> None:
    """One include of the shared set — not a local rules/ dir, not a per-study copy."""
    snakefile = (study / "Snakefile").read_text()
    assert 'include: str(rules_dir() / "study.smk")' in snakefile
    assert "from neofoam.tooling.workflow.rules import rules_dir" in snakefile
    assert not (study / "rules").exists(), f"{study.name} has its own rules/ again"


@pytest.mark.parametrize("study", _studies(), ids=lambda p: p.name)
def test_a_study_carries_no_pipeline_source(study: Path) -> None:
    """No .smk beside a study, and nothing reaching for the deleted package."""
    assert not list(study.rglob("*.smk"))
    for path in (study / "Snakefile", study / "discover.py", *study.glob("*.yaml")):
        assert "tooling.verification" not in path.read_text(), path
