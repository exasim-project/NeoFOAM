# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

import sys

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
def incompressiblefluid(ctx: typer.Context) -> None:
    """incompressibleFluid - General incompressible fluid solver with turbulence models."""

    from neofoam.solver.incompressibleFluid import run

    # Only pass the extra args (not the Typer command path)
    argv = [sys.argv[0]] + [str(arg) for arg in ctx.args]

    run(argv)


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
def incompressiblefluidneon(ctx: typer.Context) -> None:
    """NeoN-backed incompressible fluid solver (PISO, laminar)."""
    from neofoam.solver.incompressibleFluidNeon import run

    argv = [sys.argv[0]] + [str(arg) for arg in ctx.args]

    run(argv, disable_fpe=True)


if __name__ == "__main__":
    app()
