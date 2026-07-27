# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

import sys
from pathlib import Path
from typing import Optional

import typer

from neofoam.agent import build_case_agent, fill_case, write_wizard_notebook
from neofoam.solver.icofoam import IcoFoam
from neofoam.solver.incompressibleFluid import run as run_incompressible_fluid
from neofoam.solver.incompressibleFluidNeoN import run as run_neon
from neofoam.solver.incompressibleVoF import run as run_incompressible_vof
from neofoam.solver.neoIcoFoam import NeoIcoFoam
from neofoam.solver.neoPimpleFoam import NeoPimpleFoam
from neofoam.solver.pimplefoam import PimpleFoam
from neofoam.telemetry.report import write_chrome_trace, write_summary_plot

app = typer.Typer()

#: Message printed when a solver command is invoked with ``-postProcess``. Kept in
#: sync by hand with ``verification.dropin.execute.POSTPROCESS_NOT_IMPLEMENTED``
#: (not imported — ``cli`` and ``tooling.workflow`` are deliberately not coupled),
#: which the drop-in study harness matches to classify a run that hit this as
#: UNSUPPORTED_CASE rather than a solver crash.
_POSTPROCESS_NOT_IMPLEMENTED = "solver -postProcess mode not implemented"


def _reject_postprocess(argv: list[str]) -> None:
    """Fail fast and clearly on ``-postProcess`` instead of falling through to
    pybFoam's ``argList``, which does not know the flag and prints a confusing
    raw usage dump. No neofoam solver implements OpenFOAM's post-processing mode
    (running the case's registered function objects without solving) — pybFoam
    exposes no functionObject-execution binding to build it on.
    """
    if "-postProcess" in argv:
        typer.echo(f"neofoam: {_POSTPROCESS_NOT_IMPLEMENTED}", err=True)
        raise typer.Exit(code=1)


# Solver command group
solver_app = typer.Typer()

app.add_typer(solver_app, name="solver", help="Run foam solvers.")

# Agent command group
agent_app = typer.Typer()

app.add_typer(agent_app, name="agent", help="LLM-driven case scaffolding.")

# MCP server command group
mcp_app = typer.Typer()

app.add_typer(mcp_app, name="mcp", help="Run the NeoFOAM MCP server.")

# Telemetry visualization command group
telemetry_app = typer.Typer()

app.add_typer(telemetry_app, name="telemetry", help="Visualize solver telemetry traces.")


@telemetry_app.command("trace")
def telemetry_trace(
    case: Path = typer.Argument(
        ..., help="Case directory (or its telemetry/ dir) holding rank*.spans.jsonl."
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file (default: <telemetry>/trace.json)."
    ),
) -> None:
    """Export the trace as a Chrome Trace-Event ``trace.json``.

    Open the file in https://ui.perfetto.dev or ``chrome://tracing`` for a
    zoomable Gantt/flame timeline (one process row per MPI rank).
    """
    written = write_chrome_trace(case, output)
    typer.echo(f"Wrote {written}")
    typer.echo("Open it in https://ui.perfetto.dev or chrome://tracing")


@telemetry_app.command("plot")
def telemetry_plot(
    case: Path = typer.Argument(
        ..., help="Case directory (or its telemetry/ dir) holding rank*.summary.json."
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output image (default: <telemetry>/summary.png)."
    ),
    top: Optional[int] = typer.Option(None, "--top", help="Keep only the N slowest operations."),
    rank: Optional[int] = typer.Option(
        None, "--rank", help="Which rank's summary to plot (default: lowest)."
    ),
) -> None:
    """Render the per-operation total wall-clock as a bar-chart image (PNG)."""
    written = write_summary_plot(case, output, top=top, rank=rank)
    typer.echo(f"Wrote {written}")


@mcp_app.command("serve")
def mcp_serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host."),
    port: int = typer.Option(8000, "--port", help="Bind port."),
    root: Optional[str] = typer.Option(
        None,
        "--root",
        help=(
            "Confine every case path under this directory (relative paths only; "
            "escapes are rejected). Recommended when the server is reachable by an "
            "untrusted client. Omit only for trusted/local use, where absolute paths "
            "are allowed."
        ),
    ),
    reload: bool = typer.Option(
        False,
        "--reload",
        help=(
            "Restart the server when the neofoam source changes (dev only; needs an "
            "editable install). Off by default."
        ),
    ),
) -> None:
    """Start the NeoFOAM MCP server (blocking, one process).

    The server is solver-agnostic: each tool takes a ``solver`` argument
    (default ``incompressibleFluid``) resolved per call.
    """
    # noqa below: mcp.app imports fastapi unguarded, so keep this optional [mcp]
    # dep out of module-import time (CLI must import without the mcp extra).
    from neofoam.mcp.app import serve  # noqa: PLC0415

    serve(host=host, port=port, root=root, reload=reload)


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
    name: str = typer.Option("case_wizard.py", "--name", help="Notebook file name to write."),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing notebook file."),
) -> None:
    """Scaffold a marimo *case wizard* notebook into TARGET.

    Writes a self-contained notebook that renders the ``incompressibleFluid``
    configs as forms (Models / Schemes / BCs / Initial values) with an AI chat
    that fills and saves the case. The notebook operates on the directory it is
    written to, so run it from there::

        neofoam agent wizard my_case
        marimo edit my_case/case_wizard.py
    """
    try:
        path = write_wizard_notebook(target, filename=name, force=force)
    except FileExistsError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Wrote case wizard to {path}")
    typer.echo(f"Run it with:  marimo edit {path}")


@solver_app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def icofoam(ctx: typer.Context) -> None:
    """Transient solver for incompressible, laminar flow of Newtonian fluids."""
    _reject_postprocess(ctx.args)
    # Only pass the extra args (not the Typer command path)
    argv = [sys.argv[0]] + [str(arg) for arg in ctx.args]

    icoFoam = IcoFoam(argv)
    icoFoam.run()


@solver_app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def pimplefoam(ctx: typer.Context) -> None:
    """Transient solver for incompressible, turbulent flow of Newtonian fluids"""
    _reject_postprocess(ctx.args)

    # Only pass the extra args (not the Typer command path)
    argv = [sys.argv[0]] + [str(arg) for arg in ctx.args]

    pimpleFoam = PimpleFoam(argv)
    pimpleFoam.run()


@solver_app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def neoicofoam(ctx: typer.Context) -> None:
    """Transient solver for incompressible, laminar flow using NeoN bindings."""
    _reject_postprocess(ctx.args)
    argv = [sys.argv[0]] + [str(arg) for arg in ctx.args]

    solver = NeoIcoFoam(argv)
    solver.run()


@solver_app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def neopimplefoam(ctx: typer.Context) -> None:
    """Transient incompressible PIMPLE solver using NeoN bindings."""
    _reject_postprocess(ctx.args)
    argv = [sys.argv[0]] + [str(arg) for arg in ctx.args]

    solver = NeoPimpleFoam(argv)
    solver.run()


@solver_app.command(
    name="incompressiblefluid",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def incompressiblefluid(ctx: typer.Context) -> None:
    """Transient PIMPLE solver for incompressible Newtonian flow."""
    _reject_postprocess(ctx.args)
    argv = [sys.argv[0]] + [str(arg) for arg in ctx.args]
    run_incompressible_fluid(argv)


@solver_app.command(
    name="incompressiblevof",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def incompressiblevof(ctx: typer.Context) -> None:
    """incompressibleVoF - interFoam-style VoF solver with surface tension and gravity."""
    _reject_postprocess(ctx.args)
    argv = [sys.argv[0]] + [str(arg) for arg in ctx.args]
    run_incompressible_vof(argv)


@solver_app.command(
    name="incompressiblefluidneon",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def incompressiblefluidneon(ctx: typer.Context) -> None:
    """Transient PIMPLE solver for incompressible flow (NeoN backend)."""
    _reject_postprocess(ctx.args)
    argv = [sys.argv[0]] + [str(arg) for arg in ctx.args]
    run_neon(argv)


@app.command()
def preprocess(case: Path) -> None:
    """Run ONLY the mesh preprocessing pipeline for <case> and stop (no time loop).

    <case> may be any path (absolute or relative to the cwd); the pipeline runs
    from inside the case, so it need not be the working directory.
    """
    # Imported at call time so tests can monkeypatch neofoam.tools.run.run_preprocess.
    from neofoam.tools.run import run_preprocess  # noqa: PLC0415

    run_preprocess([sys.argv[0], "-case", str(case)])


if __name__ == "__main__":
    app()
