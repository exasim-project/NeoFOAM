import sys

import typer

app = typer.Typer()

# Solver command group
solver_app = typer.Typer()

app.add_typer(solver_app, name="solver", help="Run foam solvers.")


@solver_app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def icofoam(ctx: typer.Context):
    """Transient solver for incompressible, laminar flow of Newtonian fluids."""
    from foamadapter.solver.icofoam import IcoFoam

    # Only pass the extra args (not the Typer command path)
    argv = [sys.argv[0]] + [str(arg) for arg in ctx.args]

    icoFoam = IcoFoam(argv)
    icoFoam.run()


@solver_app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def pimplefoam(ctx: typer.Context):
    """Transient solver for incompressible, turbulent flow of Newtonian fluids"""

    from foamadapter.solver.pimplefoam import PimpleFoam

    # Only pass the extra args (not the Typer command path)
    argv = [sys.argv[0]] + [str(arg) for arg in ctx.args]

    pimpleFoam = PimpleFoam(argv)
    pimpleFoam.run()


if __name__ == "__main__":
    app()
