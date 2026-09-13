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

Executor selection and GPU initialization
-----------------------------------------

The NeoN-backed solvers place their fields on the executor named by the
``executor`` entry of ``system/controlDict`` — the same entry the C++ solvers
read:

.. code-block:: cpp

    executor        GPU;   // Serial (default) | CPU | GPU | default

A case without the entry runs on ``Serial``, so a stock ``pimpleFoam`` case
needs no NeoN keys.

The entry selects where the *data* is placed, not which Kokkos backends come
up. ``Kokkos::initialize`` brings up every backend the library was compiled
with, so a CUDA-enabled build takes a CUDA context per rank (a few hundred MiB)
even for a ``Serial`` or ``CPU`` run — the ``executor`` entry cannot switch that
off. When that context cannot be created, the solver aborts with a
``NeoN (Kokkos) initialization failed on the GPU backend`` error. Ways out:

* free the device, or select another one:
  ``KOKKOS_VISIBLE_DEVICES=<index>`` (``nvidia-smi`` lists the indices and
  their free memory);
* run fewer ranks per device — every rank takes its own context;
* for a host-only run, use a NeoFOAM built with ``-DKokkos_ENABLE_CUDA=OFF``.

Preprocessing — ``neofoam preprocess``
--------------------------------------

Run only the mesh preprocessing pipeline for a case and stop (no time loop):

.. code-block:: bash

    neofoam preprocess my_case

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
