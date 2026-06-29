# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

import sys
from pathlib import Path

import typer

app = typer.Typer()

# Solver command group
solver_app = typer.Typer()

app.add_typer(solver_app, name="solver", help="Run foam solvers.")


@solver_app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def icofoam(ctx: typer.Context) -> None:
    """Transient solver for incompressible, laminar flow of Newtonian fluids."""
    from neofoam.solver.icofoam import IcoFoam

    # Only pass the extra args (not the Typer command path)
    argv = [sys.argv[0]] + [str(arg) for arg in ctx.args]

    icoFoam = IcoFoam(argv)
    icoFoam.run()


@solver_app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def pimplefoam(ctx: typer.Context) -> None:
    """Transient solver for incompressible, turbulent flow of Newtonian fluids"""

    from neofoam.solver.pimplefoam import PimpleFoam

    # Only pass the extra args (not the Typer command path)
    argv = [sys.argv[0]] + [str(arg) for arg in ctx.args]

    pimpleFoam = PimpleFoam(argv)
    pimpleFoam.run()


@solver_app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def neoicofoam(ctx: typer.Context) -> None:
    """Transient solver for incompressible, laminar flow using NeoN bindings."""
    from neofoam.solver.neoIcoFoam import NeoIcoFoam

    argv = [sys.argv[0]] + [str(arg) for arg in ctx.args]

    solver = NeoIcoFoam(argv)
    solver.run()


@solver_app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def neopimplefoam(ctx: typer.Context) -> None:
    """Transient incompressible PIMPLE solver using NeoN bindings."""
    from neofoam.solver.neoPimpleFoam import NeoPimpleFoam

    argv = [sys.argv[0]] + [str(arg) for arg in ctx.args]

    solver = NeoPimpleFoam(argv)
    solver.run()


@solver_app.command(
    name="incompressiblefluid",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def incompressiblefluid(ctx: typer.Context) -> None:
    """Transient PIMPLE solver for incompressible Newtonian flow."""
    from neofoam.solver.incompressibleFluid import run as run_incompressible_fluid

    argv = [sys.argv[0]] + [str(arg) for arg in ctx.args]
    run_incompressible_fluid(argv)


@app.command()
def preprocess(case: Path) -> None:
    """Run ONLY the mesh preprocessing pipeline for <case> and stop (no time loop)."""
    from neofoam.tools.run import run_preprocess

    run_preprocess([sys.argv[0], "-case", str(case)])


if __name__ == "__main__":
    app()
