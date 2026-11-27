# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""A plugin command group whose ``hello`` command records the name it is called with."""

import typer

calls: list[str] = []

app = typer.Typer()


@app.command()
def hello(name: str) -> None:
    """Record ``name`` instead of doing work."""
    calls.append(name)


@app.command()
def bye() -> None:
    """A second command, so ``hello`` stays a subcommand of the group."""


#: Not a Typer group: an entry point naming it must be rejected.
not_a_group = hello
