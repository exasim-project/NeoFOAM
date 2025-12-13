# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

import typer


def hello() -> None:
    """Say hello from foamadapter CLI."""
    typer.echo("Hello, FoamAdapter user!")


app = typer.Typer()
app.command()(hello)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
