The ``neofoam`` command line
============================

The package installs a single Typer entry point, ``neofoam``
(``src/neofoam/cli/app.py``). Run ``neofoam --help`` — or ``--help`` on any
subcommand — for the authoritative option list.

Solvers — ``neofoam solver``
----------------------------

Run a foam solver on a case. Extra arguments are passed through to the solver
unchanged (OpenFOAM-style, e.g. ``-case <dir>``):

.. code-block:: bash

    neofoam solver icofoam -case my_case
    neofoam solver incompressiblefluid -case my_case

Available solvers: ``icofoam``, ``pimplefoam``, ``incompressiblefluid``,
``incompressiblevof`` (pybFoam backend); ``neoicofoam``, ``neopimplefoam``,
``incompressiblefluidneon`` (NeoN backend).

Preprocessing — ``neofoam preprocess``
--------------------------------------

Run only the mesh preprocessing pipeline for a case and stop (no time loop):

.. code-block:: bash

    neofoam preprocess my_case

Case scaffolding — ``neofoam agent``
------------------------------------

LLM-driven case scaffolding (needs the ``agent`` extra):

.. code-block:: bash

    neofoam agent fill src_case new_case    # fill a new case from a source case
    neofoam agent fill src_case new_case --no-llm   # forms only, no LLM call
    neofoam agent wizard .                  # write the marimo wizard notebook

MCP server — ``neofoam mcp``
----------------------------

Start the NeoFOAM MCP server (needs the ``mcp`` extra):

.. code-block:: bash

    neofoam mcp serve                # bind 127.0.0.1:8000
    neofoam mcp serve --root cases/  # confine case paths for untrusted clients

Telemetry — ``neofoam telemetry``
---------------------------------

Visualize solver telemetry traces (needs the ``telemetry`` extra):

.. code-block:: bash

    neofoam telemetry trace my_case  # Chrome trace.json for ui.perfetto.dev
    neofoam telemetry plot my_case   # per-operation wall-clock bar chart (PNG)
