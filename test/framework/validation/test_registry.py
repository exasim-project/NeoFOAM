# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

import subprocess
import sys
from pathlib import Path

from neofoam.framework.validation import (
    CaseContext,
    CheckRegistry,
    Finding,
    ValidationReport,
)


def _ctx() -> CaseContext:
    return CaseContext(case=Path("/nonexistent"), solver=object())


def test_run_preserves_registration_order() -> None:
    reg = CheckRegistry()
    reg.add("a", lambda ctx: [Finding(level="warning", file="a", message="a")])
    reg.add("b", lambda ctx: [Finding(level="warning", file="b", message="b")])
    report = reg.run(_ctx())
    assert [f.file for f in report.findings] == ["a", "b"]


def test_run_ok_true_only_when_no_error_finding() -> None:
    reg = CheckRegistry()
    reg.add("warn", lambda ctx: [Finding(level="warning", file="x", message="m")])
    assert reg.run(_ctx()).ok is True


def test_run_reports_ok_false_on_any_error_finding() -> None:
    reg = CheckRegistry()
    reg.add("bad", lambda ctx: [Finding(level="error", file="x", message="m")])
    assert reg.run(_ctx()).ok is False


def _boom(ctx: CaseContext) -> list[Finding]:
    raise RuntimeError("backend exploded")


def test_run_reports_ok_false_when_a_check_raises() -> None:
    # "every check ran" is structural: a check that cannot run is an error finding,
    # never a swallowed skip that lets ok stay True.
    reg = CheckRegistry()
    reg.add("exploder", _boom)
    report = reg.run(_ctx())
    assert report.ok is False
    hit = next(f for f in report.findings if "exploder" in f.message)
    assert hit.level == "error" and "backend exploded" in hit.message


def test_run_tags_a_raising_check_with_a_sentinel_file() -> None:
    reg = CheckRegistry()
    reg.add("exploder", _boom)
    report = reg.run(_ctx())
    hit = next(f for f in report.findings if "exploder" in f.message)
    assert hit.file == "<check:exploder>"


def test_finding_and_report_round_trip_json() -> None:
    report = ValidationReport(
        ok=False,
        findings=[Finding(level="error", file="0/U", message="m", fix="do x")],
    )
    assert "0/U" in report.model_dump_json()


def test_validation_package_imports_without_mcp() -> None:
    # pybFoam is a hard dependency the eager ``neofoam`` import always pulls; only the
    # optional mcp extra must stay unloaded when importing the validation package.
    code = (
        "import sys, neofoam.framework.validation\n"
        "assert not any(m.startswith('neofoam.mcp') for m in sys.modules), 'mcp leaked'\n"
        "assert 'fastmcp' not in sys.modules, 'fastmcp leaked'\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
