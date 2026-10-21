Pre-processing
==============

A case declares a **pipeline of tools** in ``system/preprocess.yaml``;
``neofoam preprocess <case>`` resolves it against the registered tools and runs
it in-process — no solver, no fields, no time loop. Order comes from
``depends_on``, not list position, and a tool that declares one dependency
receives that dependency's mesh:

.. code-block:: yaml

    tools:
      - tool: blockMesh
      - tool: setFields
        depends_on: [blockMesh]

The tools shipped with NeoFOAM are ``blockMesh`` and ``snappyHexMesh`` (build the
mesh from ``system/blockMeshDict`` / ``system/snappyHexMeshDict``), ``checkMesh``
(validate it, optionally failing the run) and ``setFields`` below.

Over MCP, the pipeline is authored with
``save_preprocess(case_dir, preprocess, set_fields=None)``: it validates every
``tools`` entry against the registered tools (an unknown tool name, a bad step
option or a ``depends_on`` naming a tool the pipeline does not declare is
rejected, naming the entry's position) and writes ``system/preprocess.yaml``; a
``set_fields`` payload is validated the same way — its regions resolved against
the ``Node`` selectors — and written as ``system/setFields.yaml``, but only when
an entry actually runs the ``setFields`` tool, since otherwise nothing would ever
read the file. Both files are **replaced, not merged**, so a call that adds a tool
resends the whole pipeline. The MCP server never runs the pipeline — that stays
``neofoam preprocess <case>``.

Initialising fields — ``setFields``
-----------------------------------

``setFields`` is the Python replacement for OpenFOAM's ``setFieldsDict``: it
writes a default value over a field's whole internal field, then a value over the
cells each declared region selects. The regions are the **post-processing**
selectors — ``box``, ``sphere``, ``not``, ``binary`` and any selector a case
registers itself — so one region language serves both ends of a case; see
:doc:`postprocessing` for what each selector takes.

The fields have to exist: the tool writes values into the ``0/<name>`` files the
case already has (``0/alpha.water``, ``0/U`` …) and never creates one. And
nothing runs unless the case declares something — with the tool listed in
``system/preprocess.yaml`` but neither ``system/setFields.yaml`` nor
``system/setFields.py`` present, it raises rather than silently doing nothing.

The declarative front door
~~~~~~~~~~~~~~~~~~~~~~~~~~

``system/setFields.yaml`` (``.yml`` and ``.json`` are read too — the suffix picks
the backend):

.. code-block:: yaml

    defaults:              # applied to the whole internal field first
      alpha.water: 0
    regions:
      - region: {type: box, min: [0, 0, -1], max: [0.1461, 0.292, 1]}
        values: {alpha.water: 1}
      - region: {type: binary, op: or,
                 left:  {type: sphere, center: [0, 0, 0], radius: 0.25},
                 right: {type: box, min: [0.35, -1, 0], max: [1.2, 1, 0.124]}}
        values: {alpha.water: 1, U: [0.5, 0, 0]}

``defaults`` is written first, then each entry of ``regions`` in order, so where
two regions overlap the later one wins.

The script front door
~~~~~~~~~~~~~~~~~~~~~

``system/setFields.py`` builds the same declaration in Python, where ``&``, ``|``
and ``~`` compose regions:

.. code-block:: python

    from neofoam.postprocess import Box, Sphere
    from neofoam.preprocess import SetFields

    setFields = SetFields(defaults={"alpha.water": 0.0})
    setFields.assign(
        Box(min=(0, 0, -1), max=(0.1461, 0.292, 1)) | Sphere(center=(0, 0, 0), radius=0.25),
        {"alpha.water": 1.0},
    )

The script defines exactly one module-level ``SetFields``. A case may use both
front doors: the script is executed first — so a selector it registers with
``@Node.register`` is selectable by ``type`` from the spec file — and the spec
file's regions are applied after the script's, so where they overlap the spec
file wins; a default declared by both is the spec file's.

A custom selector is defined in **one** of the case's scripts only. A case that
registers the same ``type`` in both ``system/setFields.py`` and
``system/postProcess.py`` makes the second one loaded ambiguous — the ``Node``
family then has two classes answering to that discriminator and the load fails.
Define it once; the other case file (and both spec files) can still use it by
``type``.

The value picks the field type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A **scalar** value sets a ``volScalarField`` and a **3-vector** value a
``volVectorField``. The class on disk is checked against the value *before* the
field is read, so setting ``U`` to ``1`` is a named error rather than an
OpenFOAM fatal error that takes the process down; so is setting one field to a
scalar in one place and to a vector in another.

Pre-processing only, pybFoam only
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``setFields`` overwrites what the time directory holds, so it belongs in the
pipeline that *prepares* a case: list it in ``system/preprocess.yaml`` and run
``neofoam preprocess`` before the solver. The ``incompressibleFluid`` solver
resolves the same file again at start, so the rule is **enforced, not just
documented**: the step applies only when the run starts at the case's earliest
time directory. A run continued from a later one — ``startFrom latestTime``, or a
``startTime`` with an older time directory beside it — logs

.. code-block:: text

    preprocess.setFields: skipped — the run starts at 0.1, which is not the case's
    earliest time directory, so this is a restart and the fields are left as they are

and passes the mesh on untouched, so a restart never re-initialises the fields it
is restarting from. (OpenFOAM's ``Time::timeIndex()`` and ``startTimeIndex()``
cannot be used for this — both read 0 at a restart as well.)

Because the ``incompressibleFluid`` solver re-resolves the pipeline, a
``setFields``-only pipeline has to say where its mesh comes from. ``neofoam
preprocess`` hands the tool the mesh it reads off disk, but the solver builds the
pipeline itself and refuses a mesh-consuming tool that heads the graph, with
*"consume a mesh but declare no depends_on and no mesh source is available"*.
Either give the entry a ``depends_on`` on a mesh-creating tool
(``depends_on: [blockMesh]``), or start the solver with ``--no-preprocess`` once
``neofoam preprocess`` has already run.

It is also **pybFoam-only**. A pybFoam volume field hands its cell values over as
a writable, zero-copy numpy view of the OpenFOAM memory, which is what makes the
write-back a few lines of numpy; a NeoN field keeps its values on an executor
that may be a device and offers no such view, so a NeoN case has no write path —
run ``neofoam preprocess`` on it with the pybFoam bindings, then solve.
