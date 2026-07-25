# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The packaged study rule set holds together as one pipeline.

``study.smk`` *is* the named, reusable set: a study Snakefile includes it and gets
the header globals plus five rules. Nothing but Snakemake reads these files, and
Snakemake only reads them when a study is actually run — a typo'd include or a
global a rule consumes but nobody defines surfaces as a parse error hours into a
sweep. These checks are the cheap version of that, run on every ``pytest``.

Deliberately no snakemake here: the workflow is an optional extra, and a dry run
mutates the study directory it is pointed at. The DAG check stays a manual
command (``snakemake --configfile config.yaml -n``), documented in the study's
Snakefile.

``test/tooling/workflow/test_rules.py`` covers the mesh sweep's registry and is
untouched by this file — the study set is deliberately outside that registry.
"""

from __future__ import annotations

import re

import pytest

from neofoam.tooling.workflow.rules import rules_dir

#: `include: str(rules_dir() / "<name>")` — how study.smk names its rule files.
_INCLUDE = re.compile(r'include:\s*str\(rules_dir\(\)\s*/\s*"([^"]+)"\)')

#: The globals study.smk defines for the rule bodies. Each rule file declares the
#: subset it consumes in a trailing `# Consumes header globals: A, B, C.` line.
_CONSUMES = re.compile(r"#\s*Consumes header globals:\s*([^.]+)\.")


def _study_smk() -> str:
    return (rules_dir() / "study.smk").read_text()


def _included_names() -> list[str]:
    """The rule files study.smk includes, parsed from the file itself.

    Parsed rather than duplicated as a tuple here: a list in the test would just be
    a second place to forget, and the point is to check what study.smk actually says.
    """
    return _INCLUDE.findall(_study_smk())


def test_study_smk_includes_the_five_pipeline_rules() -> None:
    assert _included_names() == [
        "study_build_case.smk",
        "study_swap_solver.smk",
        "study_run.smk",
        "study_compare.smk",
        "study_report.smk",
    ]


@pytest.mark.parametrize("name", _included_names())
def test_each_included_file_ships_and_defines_its_rule(name: str) -> None:
    """Every include resolves under rules_dir() and holds the rule it is named for.

    Catches the failure mode of a packaged rule set: a file that was never added to
    the package (so ``rules_dir()`` cannot find it) or renamed on one side only.
    """
    path = rules_dir() / name
    assert path.is_file(), f"{name} is included by study.smk but not packaged"
    rule = name.removeprefix("study_").removesuffix(".smk")
    assert re.search(rf"^rule {re.escape(rule)}:$", path.read_text(), re.M)


@pytest.mark.parametrize("name", _included_names())
def test_every_header_global_a_rule_consumes_is_defined_by_study_smk(name: str) -> None:
    """The contract between the header and the rule bodies, checked both ways.

    A rule reaching for a global ``study.smk`` does not define is a NameError at
    parse time; the declaration comment is only worth writing if it is true.
    """
    body = (rules_dir() / name).read_text()
    declared = _CONSUMES.search(body)
    assert declared, f"{name} does not declare the header globals it consumes"

    header = _study_smk()
    for symbol in (s.strip() for s in declared.group(1).split(",")):
        assert re.search(rf"^{re.escape(symbol)} = ", header, re.M), (
            f"{name} consumes {symbol}, which study.smk does not define"
        )


@pytest.mark.parametrize("name", _included_names())
def test_rule_bodies_shell_out_to_the_study_runner(name: str) -> None:
    """Every ``python -m`` in the set names the one worker CLI these rules share.

    ``study_run.smk`` is plain shell (``./Allrun``) and so names no module; every
    other rule must reach the runner at its current path, not a stale one.
    """
    body = (rules_dir() / name).read_text()
    modules = re.findall(r"python -m ([\w.]+)", body)
    assert all(m == "neofoam.tooling.workflow.study.runner" for m in modules), modules


def test_study_smk_defines_the_report_the_study_all_rule_targets() -> None:
    """``rule all: input: REPORT`` in a study Snakefile resolves to the sink's output."""
    assert re.search(r"^REPORT = ", _study_smk(), re.M)
    assert re.search(r"^\s+REPORT,$", (rules_dir() / "study_report.smk").read_text(), re.M)
