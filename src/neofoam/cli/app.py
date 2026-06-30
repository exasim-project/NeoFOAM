# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

import sys
from typing import Optional
from pathlib import Path

import typer

app = typer.Typer()

# Solver command group
solver_app = typer.Typer()

app.add_typer(solver_app, name="solver", help="Run foam solvers.")

# Agent command group
agent_app = typer.Typer()

app.add_typer(agent_app, name="agent", help="LLM-driven case scaffolding.")

# MCP server command group
mcp_app = typer.Typer()

app.add_typer(mcp_app, name="mcp", help="Run the NeoFOAM MCP server.")


@mcp_app.command("serve")
def mcp_serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host."),
    port: int = typer.Option(8000, "--port", help="Bind port."),
) -> None:
    """Start the NeoFOAM MCP server (blocking, one process).

    The server is solver-agnostic: each tool takes a ``solver`` argument
    (default ``incompressibleFluid``) resolved per call.
    """
    from neofoam.mcp.app import serve

    serve(host=host, port=port)


@agent_app.command("fill")
def agent_fill(
    source: str = typer.Argument(..., help="Source case directory."),
    target: str = typer.Argument(..., help="Target case directory."),
    no_llm: bool = typer.Option(
        False,
        "--no-llm",
        help=(
            "Skip the LLM and roundtrip the source configs from disk. Useful"
            " for testing the schema + IO path without an API key."
        ),
    ),
    model_name: str = typer.Option(
        "claude-haiku-4-5",
        "--model",
        help="Anthropic model name when not in --no-llm mode.",
    ),
    prompt: Optional[str] = typer.Option(
        None,
        "--prompt",
        help=(
            "Extra natural-language guidance appended to the source-case"
            " text before sending to the LLM."
        ),
    ),
) -> None:
    """Fill the configs of TARGET by reading SOURCE.

    Copies mesh + ``0/``/``0.orig/`` fields from SOURCE, then either
    asks an LLM to rewrite the configs (default) or roundtrips them from
    disk (``--no-llm``). Either way TARGET ends up runnable by
    ``neofoam solver incompressiblefluid``.
    """
    from neofoam.agent import build_case_agent, fill_case

    agent_obj = None
    if not no_llm:
        agent_obj = build_case_agent(model_name=model_name)
        if prompt:
            # The default prompt only contains source-case text; append the
            # user note as an extra instruction the agent will see verbatim.
            original_run = agent_obj.run_sync

            def _run_sync(p: str, *args: object, **kwargs: object) -> object:
                return original_run(f"{p}\n\nAdditional guidance:\n{prompt}")

            agent_obj.run_sync = _run_sync

    spec = fill_case(source, target, agent=agent_obj)
    written = [n for n in type(spec).model_fields if getattr(spec, n) is not None]
    typer.echo(f"Wrote configs to {target}: {', '.join(written)}")


@agent_app.command("wizard")
def agent_wizard(
    target: str = typer.Argument(
        ".", help="Directory to scaffold the wizard notebook into (default: cwd)."
    ),
    name: str = typer.Option(
        "case_wizard.py", "--name", help="Notebook file name to write."
    ),
    force: bool = typer.Option(
        False, "--force", help="Overwrite an existing notebook file."
    ),
) -> None:
    """Scaffold a marimo *case wizard* notebook into TARGET.

    Writes a self-contained notebook that renders the ``incompressibleFluid``
    configs as forms (Models / Schemes / BCs / Initial values) with an AI chat
    that fills and saves the case. The notebook operates on the directory it is
    written to, so run it from there::

        neofoam agent wizard my_case
        marimo edit my_case/case_wizard.py
    """
    from neofoam.agent import write_wizard_notebook

    try:
        path = write_wizard_notebook(target, filename=name, force=force)
    except FileExistsError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Wrote case wizard to {path}")
    typer.echo(f"Run it with:  marimo edit {path}")


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
