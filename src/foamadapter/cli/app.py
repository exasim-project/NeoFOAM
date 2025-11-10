import typer
import sys

app = typer.Typer()

# Solver command group
solver_app = typer.Typer()

app.add_typer(solver_app, name="solver", help="Run foam solvers.")


@solver_app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def icofoam(ctx: typer.Context):
    """Transient solver for incompressible, laminar flow of Newtonian fluids."""
    from foamadapter.solver.icofoam.icofoam import IcoFoam

    # Only pass the extra args (not the Typer command path)
    argv = [sys.argv[0]] + [str(arg) for arg in ctx.args]

    icoFoam = IcoFoam(argv)
    icoFoam.run()


if __name__ == "__main__":
    app()
