# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""``SolverSpec.instantiate`` always hands OpenFOAM a program name in ``argv[0]``."""

from neofoam.framework.solver import Solver


def test_instantiate_empty_argv_gets_a_program_name() -> None:
    assert Solver("emptyArgv").instantiate().argv == ["neofoam"]
    assert Solver("emptyArgv").instantiate(argv=[]).argv == ["neofoam"]


def test_instantiate_keeps_a_given_argv() -> None:
    argv = ["prog", "-case", "x"]

    assert Solver("givenArgv").instantiate(argv=argv).argv == argv
