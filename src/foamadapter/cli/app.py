# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

import sys

import typer
import pybFoam as pyf

app = typer.Typer()

# Solver command group
solver_app = typer.Typer()

app.add_typer(solver_app, name="solver", help="Run foam solvers.")


@solver_app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)  # type: ignore[untyped-decorator]
def icofoam(ctx: typer.Context) -> None:
    """Transient solver for incompressible, laminar flow of Newtonian fluids."""
    from foamadapter.solver.icofoam import IcoFoam

    # Only pass the extra args (not the Typer command path)
    argv = [sys.argv[0]] + [str(arg) for arg in ctx.args]

    icoFoam = IcoFoam(argv)
    icoFoam.run()


@solver_app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)  # type: ignore[untyped-decorator]
def pimplefoam(ctx: typer.Context) -> None:
    """Transient solver for incompressible, turbulent flow of Newtonian fluids"""

    from foamadapter.solver.pimplefoam import PimpleFoam

    # Only pass the extra args (not the Typer command path)
    argv = [sys.argv[0]] + [str(arg) for arg in ctx.args]

    pimpleFoam = PimpleFoam(argv)
    pimpleFoam.run()


@solver_app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)  # type: ignore[untyped-decorator]
def incompressiblefluid(ctx: typer.Context) -> None:
    """Solver for incompressible, turbulent flow of Newtonian fluids"""

    from foamadapter.solver.incompressibleFluid import IncompressibleFluid

    # Only pass the extra args (not the Typer command path)
    argv = [sys.argv[0]] + [str(arg) for arg in ctx.args]

    # Auto-detect models from case files
    models = []

    # Check if Boussinesq model should be used (beta and TRef in transportProperties)
    try:
        props = pyf.dictionary.read("constant/transportProperties")
        keys = list(props.toc())
        if "beta" in keys and "TRef" in keys:
            from foamadapter.models.buoyancy import BoussinesqModel

            models.append(BoussinesqModel())
    except Exception:
        pass  # Not a Boussinesq case

    incompressibleFluid = IncompressibleFluid(argv=argv, models=models)
    incompressibleFluid.run()


if __name__ == "__main__":
    app()
