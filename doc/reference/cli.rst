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

Plugin packages
~~~~~~~~~~~~~~~

A separately installed package extends neofoam through four entry-point groups
(``neofoam.core.entry_points``; ``neofoam.ui.steps`` is read by
``neofoam.ui.plugins``):

.. code-block:: toml

    [project.entry-points."neofoam.plugins"]
    mdao = "neofoam_mdao.models"        # module; importing it registers models

    [project.entry-points."neofoam.solvers"]
    mdaofluid = "neofoam_mdao.solver:run"   # run(argv) -> neofoam solver mdaofluid

    [project.entry-points."neofoam.region_solvers"]
    mdaosolid = "neofoam_mdao.region:mdaosolid"   # SolverSpec, found by name

    [project.entry-points."neofoam.ui.steps"]
    mdaostep = "neofoam_mdao.ui:make_step"   # factory of an extra wizard step

The CLI imports every ``neofoam.plugins`` module before running a command, so a
plugin can add to any ``PluginSystem`` family (viscosity models, post-processing
nodes, …). The sweep runner, MCP server and UI module entry points load them too.
A plugin that fails to import aborts the command. Every installed plugin module
is imported when a neofoam process starts, so install only plugin packages you
trust.

Solver commands are lower-case: an entry point ``myFluid`` runs as
``neofoam solver myfluid``, like the built-in solvers. A plugin solver whose
lower-cased name matches another solver command is refused.

A ``neofoam.region_solvers`` spec is found by the solver registry; a built-in
solver of the same name takes precedence.

Adding a new kind of model
~~~~~~~~~~~~~~~~~~~~~~~~~~

A new member of a family a solver already binds needs only a registration:

.. literalinclude:: ../../test/solver/_optional_family_plugin.py
    :language: python
    :start-after: # docs-existing-family-start
    :end-before: # docs-existing-family-end

A new *optional* family is bound to a solver with ``solver.models(family)``. The
family provides ``all_specs()`` (its members, for the config schema) and
``detect_models(case_dir)`` (the members active for a case):

.. literalinclude:: ../../test/solver/_optional_family_plugin.py
    :language: python
    :start-after: # docs-new-family-start
    :end-before: # docs-new-family-end

Limits:

* Supported by ``incompressibleFluid``, ``incompressibleFluidNeoN`` and
  ``incompressibleVoF``, whose load steps call ``detect_optional_models``. The
  script solvers (``icofoam``, ``pimplefoam``, ``neoicofoam``, ``neopimplefoam``)
  have no solver spec. A plugin solver must call it in its own load step.
* Each active member needs ``@config`` or ``@load``, or it cannot be instantiated.
* Bound families load after the solver's own optional family, in binding order.
* Operations must fit the solver's execution graph: ``depends_on`` and ``before``
  name operations of that solver. ``momentum`` exists in all three loops; most
  other names and the fields available differ per solver.
* Only optional families are extensible this way. The load steps select their
  required families directly, so ``models(family, required=True)`` adds a family
  to the schema but does not run it.

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

The pipeline and the tools it can run are described in :doc:`preprocessing`.

MCP server — ``neofoam mcp``
----------------------------

Start the NeoFOAM MCP server (needs the ``mcp`` extra):

.. code-block:: bash

    neofoam mcp serve                # bind 127.0.0.1:8000
    neofoam mcp serve --root cases/  # confine case paths for untrusted clients

Case wizard — ``neofoam ui``
----------------------------

Launch the case-wizard web UI (needs the ``ui`` extra) — see
:doc:`/tutorials/case-wizard` and :doc:`ui-architecture`:

.. code-block:: bash

    neofoam ui                              # serve http://localhost:8080/ and open a browser
    neofoam ui --solver incompressibleVoF   # edit another solver's configs
    neofoam ui --port 8081 --no-browser     # another port, no browser
    neofoam ui --host 0.0.0.0               # bind every interface (phone on the LAN)

Telemetry — ``neofoam telemetry``
---------------------------------

Visualize solver telemetry traces (needs the ``telemetry`` extra):

.. code-block:: bash

    neofoam telemetry trace my_case  # Chrome trace.json for ui.perfetto.dev
    neofoam telemetry plot my_case   # per-operation wall-clock bar chart (PNG)
