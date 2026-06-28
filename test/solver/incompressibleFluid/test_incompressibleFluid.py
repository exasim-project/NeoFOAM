# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Unit coverage for incompressibleFluid entrypoint helpers (no mesh built)."""

import pytest

pytest.importorskip("pybFoam")

from neofoam.solver.incompressibleFluid.incompressibleFluid import (  # noqa: E402
    _case_dir_from_argv,
)


def test_case_dir_from_argv_reads_case_flag() -> None:
    assert _case_dir_from_argv(["solver", "-case", "/x"]) == "/x"


def test_case_dir_from_argv_defaults_when_absent() -> None:
    assert _case_dir_from_argv(["preprocess"]) == "."


def test_case_dir_from_argv_defaults_when_flag_dangling() -> None:
    assert _case_dir_from_argv(["-case"]) == "."
