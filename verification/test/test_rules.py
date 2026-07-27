# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Each study's own rule set holds together as one pipeline.

``rules/header.smk`` is the named set *within a study* now — a study's own
Snakefile includes it and gets the header globals plus five rules. Nothing but
Snakemake reads these files, and Snakemake only reads them when a study is
actually run — a typo'd include or a global a rule consumes but nobody defines
surfaces as a parse error hours into a sweep. These checks are the cheap version
of that, run on every ``pytest``, once per study: the two copies are no longer
one shared file, so they can drift independently and each needs its own check.

``test/tooling/workflow/test_rules.py`` covers the mesh sweep's registry and is
untouched by this file — the study set was never in that registry.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_STUDIES_ROOT = Path(__file__).parents[2] / "verification" / "foam_tutorials"
_STUDIES = ("incompressibleFluid", "incompressibleVoF")
_RULE_FILES = ("build_case.smk", "swap_solver.smk", "run.smk", "compare.smk", "report.smk")

#: `include: "<name>"` — how header.smk names its rule files (relative, no rules_dir()).
_INCLUDE = re.compile(r'include:\s*"([^"]+)"')

#: The globals header.smk defines for the rule bodies. Each rule file declares the
#: subset it consumes in a trailing `# Consumes header globals: A, B, C.` line.
_CONSUMES = re.compile(r"#\s*Consumes header globals:\s*([^.]+)\.")


def _rules_dir(study: str) -> Path:
    return _STUDIES_ROOT / study / "rules"


def _header(study: str) -> str:
    return (_rules_dir(study) / "header.smk").read_text()


def _included_names(study: str) -> list[str]:
    """The rule files header.smk includes, parsed from the file itself."""
    return _INCLUDE.findall(_header(study))


@pytest.mark.parametrize("study", _STUDIES)
def test_header_smk_includes_the_five_pipeline_rules(study: str) -> None:
    assert _included_names(study) == list(_RULE_FILES)


@pytest.mark.parametrize("study", _STUDIES)
@pytest.mark.parametrize("name", _RULE_FILES)
def test_each_included_file_ships_and_defines_its_rule(study: str, name: str) -> None:
    """Every include resolves under the study's own rules/ and holds its rule.

    Catches the failure mode of a per-study rule set: a file that was renamed or
    dropped on one side (the two studies no longer share one packaged file, so
    nothing else would catch this).
    """
    path = _rules_dir(study) / name
    assert path.is_file(), f"{study}: {name} is included by header.smk but not on disk"
    rule = name.removesuffix(".smk")
    assert re.search(rf"^rule {re.escape(rule)}:$", path.read_text(), re.M)


@pytest.mark.parametrize("study", _STUDIES)
@pytest.mark.parametrize("name", _RULE_FILES)
def test_every_header_global_a_rule_consumes_is_defined_by_header_smk(
    study: str, name: str
) -> None:
    """The contract between the header and the rule bodies, checked both ways.

    A rule reaching for a global header.smk does not define is a NameError at
    parse time; the declaration comment is only worth writing if it is true.
    """
    body = (_rules_dir(study) / name).read_text()
    declared = _CONSUMES.search(body)
    assert declared, f"{study}/{name} does not declare the header globals it consumes"

    header = _header(study)
    for symbol in (s.strip() for s in declared.group(1).split(",")):
        assert re.search(rf"^{re.escape(symbol)} = ", header, re.M), (
            f"{study}/{name} consumes {symbol}, which header.smk does not define"
        )


@pytest.mark.parametrize("study", _STUDIES)
@pytest.mark.parametrize("name", _RULE_FILES)
def test_rule_bodies_shell_out_to_the_dropin_runner(study: str, name: str) -> None:
    """Every ``python -m`` in the set names the one worker CLI these rules share.

    ``run.smk`` is plain shell (``./Allrun``) and so names no module — it is still
    parametrized here so a future rule added to it that *does* shell out is caught
    by the same check, matching the fixture's pre-move behaviour.
    """
    body = (_rules_dir(study) / name).read_text()
    modules = re.findall(r"python -m ([\w.]+)", body)
    assert all(m == "verification.dropin.runner" for m in modules), (study, name, modules)


@pytest.mark.parametrize("study", _STUDIES)
def test_header_smk_defines_the_report_the_study_all_rule_targets(study: str) -> None:
    """``rule all: input: REPORT`` in a study Snakefile resolves to the sink's output."""
    assert re.search(r"^REPORT = ", _header(study), re.M)
    assert re.search(r"^\s+REPORT,$", (_rules_dir(study) / "report.smk").read_text(), re.M)
