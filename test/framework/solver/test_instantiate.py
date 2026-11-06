# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""``SolverSpec.instantiate`` always hands OpenFOAM a program name in ``argv[0]``."""

from neofoam.framework.solver import Solver


def test_empty_argv_gets_a_program_name() -> None:
    # pyf.argList([]) segfaults; run() without arguments must not reach it with [].
    assert Solver("emptyArgv").instantiate().argv == ["neofoam"]
    assert Solver("emptyArgv").instantiate(argv=[]).argv == ["neofoam"]


def test_given_argv_is_kept() -> None:
    assert Solver("givenArgv").instantiate(argv=["prog", "-case", "x"]).argv == [
        "prog",
        "-case",
        "x",
    ]
