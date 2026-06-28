# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for per-solver Tool registration on SolverSpec.

Locks in that ``tools(...)`` registers idempotently and per-solver,
``detect_preprocess_tools`` resolves the enable file against the registered
set (raising on an unregistered entry), a tool is reusable across solvers, and
``model_specs`` unions the registered tools alongside core/optional members.
"""

from pathlib import Path

import pytest

from neofoam.framework.solver import Solver
from neofoam.framework.tools import PreprocessConfig, Tool


def _enable_file(tmp_path: Path, body: str) -> Path:
    system = tmp_path / "system"
    system.mkdir()
    (system / "preprocess.yaml").write_text(body)
    return tmp_path


def test_tools_register_additive_deduped() -> None:
    a = Tool("a")
    spec = Solver("fake").tools(a)
    spec.tools(a)  # idempotent
    assert spec._tools == [a]


def test_tools_empty_without_registration() -> None:
    spec = Solver("fake")
    assert spec._tools == []
    assert spec.detect_preprocess_tools(Path(".")) == []


def test_detect_preprocess_tools_resolves_registered(tmp_path: Path) -> None:
    a = Tool("a")  # untyped build → raw mapping
    a.build(lambda cfg: [])
    case = _enable_file(tmp_path, "tools:\n  - tool: a\n")
    spec = Solver("fake").tools(a)
    rts = spec.detect_preprocess_tools(case)
    assert [rt.name for rt in rts] == ["preprocess.a"]


def test_detect_preprocess_tools_unknown_tool_raises(tmp_path: Path) -> None:
    case = _enable_file(tmp_path, "tools:\n  - tool: a\n")
    spec = Solver("fake")  # did NOT register 'a'
    with pytest.raises(ValueError, match="a"):
        spec.detect_preprocess_tools(case)


def test_same_tool_resolves_on_two_solvers(tmp_path: Path) -> None:
    a = Tool("a")
    a.build(lambda cfg: [])
    case = _enable_file(tmp_path, "tools:\n  - tool: a\n")
    s1 = Solver("one").tools(a)
    s2 = Solver("two").tools(a)
    assert [r.name for r in s1.detect_preprocess_tools(case)] == ["preprocess.a"]
    assert [r.name for r in s2.detect_preprocess_tools(case)] == ["preprocess.a"]


def test_model_specs_unions_tools() -> None:
    a = Tool("a")
    a.config(PreprocessConfig)
    spec = Solver("fake").tools(a)
    assert a in spec.model_specs
