In-situ post-processing
=======================

A case declares **tables**; the solver evaluates them while it runs and appends
each one to ``postProcessing/<name>.csv`` in the case directory, ``time`` first
column. A table is a *source* (where the numbers come from), a *pipeline* of
nodes (what happens to them), a *write control* (how often) and a *writer* (in
what format — CSV unless the table says otherwise).

Nothing runs unless the case declares something: with neither
``system/postProcess.yaml`` nor ``system/postProcess.py`` present, the
``postProcess`` model is a no-op and no ``postProcessing/`` directory is created.

The ``postProcess`` model is wired into ``incompressibleFluid``,
``incompressibleVoF`` and ``incompressibleFluidNeoN`` today —
another solver picks it up by adding the same seam to its ``create_fields.py``
(instantiate the model) and its execution-graph step (step ``post_process`` after
``write_output``); until it does, a declaration in one of its cases is ignored
silently.

On ``incompressibleFluidNeoN`` only the ``internal`` source works. Its fields
live on the NeoN executor, and the source reads them through a host copy of the
field's internal vector — a few hundred microseconds per million cells on a host
executor, a few milliseconds off a GPU, and only on the steps a table is actually
due. The cell geometry comes off the run-time adapter's mesh, which *is* an
fvMesh, so volume and surface integrals need no separate path. The other sources
interpolate a *pybFoam* field onto a sampled surface or set, and NeoN never
writes its values back into the OpenFOAM registry while the run goes on: a
``patch``, ``line``, ``plane`` or ``isoSurface`` table on a NeoN case raises
``postProcess: cannot sample '<field>'; ScalarVolumeField is not one of
['volScalarField', 'volVectorField']`` at the table's first evaluation. So does
``residuals``, which reads the fvMesh's ``solverPerformanceDict`` — NeoN solves
through Ginkgo and never fills it.

Field names follow the solver's registry, which is not always the name on disk:
``incompressibleVoF`` registers the phase fractions as ``alpha1`` and ``alpha2``
while the files are ``alpha.water`` and ``alpha.air``. The ``internal`` source
accepts either spelling — it falls back to matching the field's own OpenFOAM
name — but sources that hand a name straight to OpenFOAM's registry, such as
``isoSurface``'s ``iso_field``, need the on-disk one.

Both front doors build the same objects, so pick whichever fits: the spec file
for a fixed set of tables, the script when you want ``|`` composition or your own
node. A case may use both — the script is executed first, so the nodes it
registers are selectable from the spec file. A table name declared twice raises,
naming both origins.

The declarative front door
--------------------------

``system/postProcess.yaml`` (``.yml`` and ``.json`` are read too — the suffix
picks the backend):

.. code-block:: yaml

    tables:
      - name: volume_p                       # -> postProcessing/volume_p.csv
        source:   {type: internal, field: p}
        pipeline: [{type: volIntegrate, name: volume_p}]

      - name: volume_U_scaled
        source: {type: internal, field: U}   # a vector: one column per component
        pipeline:
          - {type: scale, factor: 1000.0}
          - {type: print, label: U_in_mm_per_s}
          - {type: volIntegrate, name: volume_U_scaled}
        write_control: {write_control_type: timeStep, interval: 10}

``source``, each ``pipeline`` entry, ``write_control`` and ``writer`` are open
mappings resolved against their plugin families by the ``type`` (respectively
``write_control_type``) key, so a newly registered node needs no schema change.
An unknown ``type`` raises, naming the table and the pipeline position, and so
does an unknown *key* inside a source, node, write control or writer — a
misspelt field is rejected on load rather than silently ignored.

The script front door
---------------------

``system/postProcess.py`` must define exactly one module-level
:class:`~neofoam.postprocess.table.TableSet`. Each decorated function is called
at import time and returns a pipeline, so a script table *is* the same
:class:`~neofoam.postprocess.node.Pipeline` a declared one resolves to — it
takes no arguments, because sources resolve against the live ``Context`` at
evaluation time.

.. code-block:: python

    from neofoam.algorithms.field_writer.write_control import IntervalWriteControl
    from neofoam.postprocess import Print, Scale, TableSet, VolIntegrate, field

    postProcess = TableSet()

    @postProcess.table("volume_p_script.csv")
    def volume_p_script():
        return field("p") | Print(label="p") | VolIntegrate(name="volume_p_script")

    @postProcess.table("volume_U_script.csv", write_control=IntervalWriteControl(interval=10))
    def volume_U_script():
        return field("U") | Scale(factor=1000.0) | VolIntegrate(name="volume_U_script")

Executing case-local Python is the same trust boundary as running the case at
all; the path is confined to the case directory.

Source catalogue
----------------

A table starts at a source: it resolves against the live simulation and hands the
pipeline the values *plus* the geometry they sit on — positions and, where one
exists, a per-element measure. Each source is selectable by its ``type`` in a spec
file and has a sugar function for a script.

.. list-table::
   :header-rows: 1
   :widths: 11 13 26 18 32

   * - ``type``
     - Class
     - Sugar
     - Geometry (measure)
     - Key fields
   * - ``internal``
     - ``InternalField``
     - ``field("p")``
     - cells (cell volume)
     - ``field``
   * - ``patch``
     - ``PatchField``
     - ``patch("p", "movingWall")``
     - patch faces (face area)
     - ``field``, ``patch``
   * - ``plane``
     - ``PlaneSurface``
     - ``plane((0.05, 0, 0), (1, 0, 0))``
     - cut faces (face area)
     - ``point``, ``normal``, ``field``, ``scheme``
   * - ``isoSurface``
     - ``IsoSurface``
     - ``iso_surface("alpha.water", 0.5)``
     - surface faces (face area)
     - ``iso_field``, ``iso_value``, ``field``, ``scheme``
   * - ``line``
     - ``LineField``
     - ``line("U", start, end, n_points)``
     - sample points (**none**)
     - ``field``, ``set_type`` and that layout's points
   * - ``residuals``
     - ``Residuals``
     - ``residuals()``
     - none — already aggregated
     -

The measure is what ``area``, ``volIntegrate`` and ``surfIntegrate`` weight with, so
a probe (no measure) rejects them: reduce it with ``mean`` or write every point with
``rows``. Sampling is interpolated onto the geometry — a patch hands back its own
boundary values, a probe point on a wall carries the boundary condition.

The two surface sources take an optional ``field``. Without one the values *are* the
face areas, so a bare surface measures its own cut; naming ``field`` (or piping into
``sample``) puts a field on it instead. ``scheme`` (default ``cellPoint``) is the
interpolation onto the faces and belongs to the surface, so every ``sample`` on it
shares it.

``set_type`` picks a probe's point layout and the rest of its spec is that layout's
data: ``uniform`` (the default) takes ``start``, ``end`` and ``n_points``, ``cloud``
and ``polyLine`` take ``points`` (``polyLine`` also an optional ``n_points``), and
``circle`` takes ``origin``, ``circle_axis``, ``start_point`` and ``d_theta``.
Missing data for the chosen layout is rejected when the table is built, not at the
first time step. The ``line()`` sugar builds the ``uniform`` case; use ``LineField``
directly for the other three.

Node catalogue
--------------

Importing ``neofoam.postprocess`` registers every node below, so its ``type``
string is valid in a spec file and its class is importable for a script.

.. list-table::
   :header-rows: 1
   :widths: 16 16 40 28

   * - ``type``
     - Class
     - What it does
     - Key fields
   * - ``box``
     - ``Box``
     - Keep the elements inside an axis-aligned box; the faces are inside.
     - ``min``, ``max``
   * - ``sphere``
     - ``Sphere``
     - Keep the elements inside a sphere; the surface is inside.
     - ``center``, ``radius``
   * - ``not``
     - ``Not``
     - Invert another selector.
     - ``region``
   * - ``binary``
     - ``Binary``
     - Combine two selectors.
     - ``op`` (``and``/``or``), ``left``, ``right``
   * - ``directional``
     - ``Directional``
     - Bin the elements by their signed distance along a direction, giving one
       row per bin (``len(bins) + 1`` of them).
     - ``bins``, ``direction``, ``origin``
   * - ``mag``
     - ``Mag``
     - Vector field to its magnitude.
     -
   * - ``component``
     - ``Component``
     - Vector field to one component.
     - ``index`` (0-2)
   * - ``area``
     - ``Area``
     - Replace the values by the per-element measure (cell volume, face area).
     -
   * - ``sample``
     - ``Sample``
     - Interpolate a registered field onto the surface the dataset already carries
       — the way to put a second field on one cut.
     - ``field``
   * - ``sum``
     - ``Sum``
     - Plain sum of the active values.
     - ``name``
   * - ``mean``
     - ``Mean``
     - Arithmetic mean of the active values (not volume weighted — for that,
       divide a ``volIntegrate`` by the ``sum`` of ``area``, which is the
       volume of the selection).
     - ``name``
   * - ``max`` / ``min``
     - ``Max`` / ``Min``
     - Extremum of the active values, component-wise for a vector.
     - ``name``
   * - ``volIntegrate``
     - ``VolIntegrate``
     - Sum weighted by the cell volume.
     - ``name``
   * - ``surfIntegrate``
     - ``SurfIntegrate``
     - Sum weighted by the face area.
     - ``name``
   * - ``rows``
     - ``Rows``
     - Terminal, but no reduction: one CSV row per active element — its bin (if
       a binner set one), its position, then its value.
     - ``name``
   * - ``scale``
     - ``Scale``
     - Multiply the values by a factor.
     - ``factor``
   * - ``print``
     - ``Print``
     - Print what flows through and pass it on unchanged.
     - ``label``

Selectors narrow the active elements and never resurrect one an earlier node
masked out; in a script they compose with ``&``, ``|`` and ``~`` (parenthesise
``a | b``, since ``|`` also chains a pipeline). A binner sets the groups an
aggregator reduces over; a table has at most one of each, in the order
*selector, binner, field function, aggregator*. ``rows`` takes the aggregator's
place when every element is wanted instead of one number.

Selecting and binning
---------------------

A region plus a binner turns a single number into a profile. In a spec file:

.. code-block:: yaml

    tables:
      - name: p_profile
        source:   {type: internal, field: p}
        pipeline:
          - {type: box, min: [0.0, 0.0, 0.0], max: [0.1, 0.1, 0.01]}
          - {type: directional, bins: [0.025, 0.05, 0.075], direction: [0, 1, 0]}
          - {type: volIntegrate, name: p_profile}

and the same table in a script:

.. code-block:: python

    from neofoam.postprocess import Box, Directional, VolIntegrate, field

    @postProcess.table("p_profile.csv")
    def p_profile():
        return (
            field("p")
            | Box(min=(0.0, 0.0, 0.0), max=(0.1, 0.1, 0.01))
            | Directional(bins=[0.025, 0.05, 0.075], direction=(0, 1, 0))
            | VolIntegrate(name="p_profile")
        )

``direction`` is normalised, so the ``bins`` are distances in metres from
``origin`` (the global origin by default) and must be strictly increasing. A
binned table writes one row per bin
on every write step, with a leading ``bin`` column holding the bin index:
``0`` below ``bins[0]``, ``len(bins)`` above ``bins[-1]``:

.. code-block:: text

    time,bin,p_profile
    0.1,0.0,0.0031
    0.1,1.0,0.0027
    0.1,2.0,0.0019
    0.1,3.0,0.0008

The row count comes from the binner's own ``bins``, not from the data, so it
does not change with the mesh or the selection.

.. note::

   A bin whose elements are all masked out has nothing to reduce, and the
   sentinel says so rather than a plausible zero: ``sum``, ``volIntegrate`` and
   ``surfIntegrate`` write ``0``, ``mean`` and ``min`` write ``+GREAT`` and
   ``max`` writes ``-GREAT``, where ``GREAT`` is OpenFOAM's ``1e15``
   (importable as ``neofoam.postprocess.GREAT``). Grep a column for ``e+15``
   before trusting a profile.

Surfaces, probes and residuals
------------------------------

One table per source family — the surface integral of ``p`` over the lid, the area
of a plane cut, the mean speed on that plane, the area of an iso-surface, a probe
along a line, and the linear solves of the step:

.. code-block:: yaml

    tables:
      - name: wall_p
        source:   {type: patch, field: p, patch: movingWall}
        pipeline: [{type: surfIntegrate, name: wall_p}]

      - name: cross_section                   # no field: the values are the areas
        source:   {type: plane, point: [0.05, 0.0, 0.0], normal: [1.0, 0.0, 0.0]}
        pipeline: [{type: sum, name: cross_section}]

      - name: mean_speed
        source:   {type: plane, point: [0.05, 0.0, 0.0], normal: [1.0, 0.0, 0.0]}
        pipeline:
          - {type: sample, field: U}
          - {type: mag}
          - {type: mean, name: mean_speed}

      - name: interface_area
        source:   {type: isoSurface, iso_field: alpha.water, iso_value: 0.5}
        pipeline: [{type: sum, name: interface_area}]

      - name: u_profile                       # one row per point, per write step
        source:
          type: line
          field: U
          start: [0.05, 0.0, 0.005]
          end:   [0.05, 0.1, 0.005]
          n_points: 20
        pipeline: [{type: mag}, {type: rows, name: magU}]

      - name: residuals                       # a whole table: no pipeline
        source: {type: residuals}

and the same six as a script:

.. code-block:: python

    from neofoam.postprocess import (
        Mag,
        Mean,
        Rows,
        Sample,
        Sum,
        SurfIntegrate,
        TableSet,
        iso_surface,
        line,
        patch,
        plane,
        residuals,
    )

    postProcess = TableSet()

    @postProcess.table("wall_p.csv")
    def wall_p():
        return patch("p", "movingWall") | SurfIntegrate(name="wall_p")

    @postProcess.table("cross_section.csv")
    def cross_section():
        return plane((0.05, 0.0, 0.0), (1.0, 0.0, 0.0)) | Sum(name="cross_section")

    @postProcess.table("mean_speed.csv")
    def mean_speed():
        cut = plane((0.05, 0.0, 0.0), (1.0, 0.0, 0.0))
        return cut | Sample(field="U") | Mag() | Mean(name="mean_speed")

    @postProcess.table("interface_area.csv")
    def interface_area():
        return iso_surface("alpha.water", 0.5) | Sum(name="interface_area")

    @postProcess.table("u_profile.csv")
    def u_profile():
        probe = line("U", (0.05, 0.0, 0.005), (0.05, 0.1, 0.005), n_points=20)
        return probe | Mag() | Rows(name="magU")

    @postProcess.table("residuals.csv")
    def solver_residuals():
        return residuals()

``rows`` writes the elements themselves: ``x``, ``y``, ``z`` — the point, face centre
or cell centre the value sits on — and then the value (``<name>_0`` to ``<name>_2``
for a vector), one line per active element on every write step. Behind a binner it
keeps the leading ``bin`` column the aggregators write, so a grouped probe stays
readable:

.. code-block:: text

    time,x,y,z,magU
    0.1,0.05,0.0025,0.005,0.0331
    0.1,0.05,0.0075,0.005,0.1042
    0.1,0.05,0.0125,0.005,0.1873

OpenFOAM drops a requested point that lies outside the mesh, and the source masks out
one it kept but could not evaluate, so a probe may write fewer rows than ``n_points``
— count the lines of a write step before reading a profile as evenly spaced.

``residuals`` aggregates itself, so nothing may be piped onto it and its columns are
fixed: one row per solve, per metric and (for a vector solve) per component.

.. code-block:: text

    time,field,solver,metric,iteration,value
    0.003,Ux,DILUPBiCGStab,initial,0,1
    0.003,Uy,DILUPBiCGStab,initial,0,1
    0.003,Uz,DILUPBiCGStab,initial,0,0
    0.003,Ux,DILUPBiCGStab,final,0,5.7e-09
    0.003,p,DICPCG,initial,0,0.0425653
    0.003,p,DICPCG,final,0,0.000471262
    0.003,p,DICPCG,initial,1,0.000563142
    0.003,p,DICPCG,final,1,3.43385e-08

``metric`` is ``initial`` or ``final``, ``iteration`` counts the solves of that field
within the step (the correctors), and a vector solve is split into OpenFOAM's ``x``,
``y``, ``z`` rows. A step that solved nothing writes no rows.

Adding a node
-------------

A node is a ``@Node.register`` subclass with a ``type: Literal[...]``
discriminator and a ``compute`` method. It sees only numpy — a
:class:`~neofoam.postprocess.node.DataSet` of ``values`` plus a ``geometry``
exposing ``positions`` and ``measure`` — and returns a *new* dataset, never
mutating its input:

.. code-block:: python

    from typing import Literal
    from neofoam.postprocess import DataSet, Node

    @Node.register
    class Clip(Node):
        """Raise every value below the threshold up to it."""

        type: Literal["clip"] = "clip"
        threshold: float = 0.0

        def compute(self, dataset: DataSet) -> DataSet:
            return dataset.with_values(dataset.values.clip(min=self.threshold))

Registering it in ``system/postProcess.py`` makes ``{type: clip, threshold: 0.0}``
valid in the same case's spec file. Put it in a package instead and it is
available to every case that imports it. The terminal node of a table is an
aggregator: it returns an
:class:`~neofoam.postprocess.node.AggregatedDataSet` — the CSV column names and
the rows to append.

A new *source* is the same pattern with ``@Source.register`` and a ``resolve(ctx)``
returning a ``DataSet``. Nodes are pure numpy; pybFoam appears in the sources,
the CSV writer (which rank owns the files) and the reductions.

Writers
-------

``writer`` picks the format one table is written in. NeoFOAM ships one,
``{type: csv}``, and that is the default, so a table only names a ``writer`` when
it wants another:

.. code-block:: yaml

    tables:
      - name: volume_p
        source:   {type: internal, field: p}
        pipeline: [{type: volIntegrate, name: volume_p}]
        writer:   {type: csv}          # the default; say nothing for the same effect

and in a script, per table or as the set's default:

.. code-block:: python

    postProcess = TableSet(writer=CsvWriter())      # every table of this set

    @postProcess.table("volume_p.csv", writer=CsvWriter())
    def volume_p():
        return field("p") | VolIntegrate(name="volume_p")

Adding a writer
~~~~~~~~~~~~~~~

A writer is a ``@TableWriter.register`` subclass with a ``type: Literal[...]``
discriminator, an ``open`` and a ``write`` — the same recipe as *Adding a node*
above, and registering it in ``system/postProcess.py`` likewise makes its
``type`` valid in the same case's spec file:

.. code-block:: python

    from pathlib import Path
    from typing import Literal

    from pydantic import PrivateAttr

    from neofoam.postprocess import AggregatedDataSet, TableWriter

    @TableWriter.register
    class TextWriter(TableWriter):
        """Write one space-separated line per row to ``<table>.txt``."""

        type: Literal["text"] = "text"

        _path: Path = PrivateAttr(default=Path())

        def open(self, path_stem: Path, *, append: bool) -> None:
            self._path = path_stem.with_name(path_stem.name + ".txt")
            self._path.parent.mkdir(parents=True, exist_ok=True)
            if not append:
                self._path.write_text("")

        def write(self, time: float, result: AggregatedDataSet) -> None:
            with self._path.open("a") as handle:
                for row in result.rows:
                    handle.write(" ".join(str(value) for value in [time, *row]) + "\n")

``open`` is handed the output path *without* a suffix — the writer appends the one
its format owns — and ``append`` says whether the run continues an earlier one.
The writer a table holds is a pure declaration; the ``postProcess`` model
deep-copies it and opens the copy, so the file state a writer keeps between calls
belongs to that copy and one declaration can serve every table of a case.

.. note::

   The class is ``TableWriter`` and not ``Writer`` because the field writer
   already owns that name
   (:class:`neofoam.algorithms.field_writer.writer.Writer`), and a plugin family
   is keyed by its class name.

Write cadence and output
------------------------

``write_control`` reuses the write policies of the field writer
(:mod:`neofoam.algorithms.field_writer.write_control`), per table:

``{write_control_type: timeStep, interval: N}``
    Every ``N`` steps — the default, with ``interval: 1``.

``{write_control_type: runTime, interval: T}``
    Every ``T`` of simulated time.

A table is only evaluated on its own write steps, so a coarse cadence costs
nothing in between. ``post_process`` is stepped **last** in the time loop, after
the fields have been written, so a CSV row and a time directory describe the same
state.

Output is flat: one ``postProcessing/<table name>`` per table, suffixed by the
writer's format (``.csv`` by default), created on the first write, ``time`` first
column. A vector-valued aggregation becomes one column
per component (``<name>_0``, ``<name>_1``, ``<name>_2``); a binned table gains a
leading ``bin`` column and one row per bin; ``rows`` puts ``x``, ``y``, ``z`` (after
that same ``bin`` column, where a binner set one) before the value; and ``residuals`` writes the string columns ``field``, ``solver`` and
``metric`` beside its numbers, so that file is long-format rather than one column
per series. A restart (``startTime > 0``) appends to an existing file; a run from
scratch truncates it.

.. note::

   **Decomposed runs.** Every rank evaluates its due tables over its own cells,
   the aggregators reduce their per-bin numbers over all ranks through pybFoam's
   collectives (``gSum`` for the additive ones, ``gMax``/``gMin`` for the
   extrema), and only the master rank's writer touches the filesystem — so a
   ``-parallel`` run leaves one set of CSVs at the case root, holding the same
   numbers a serial run of the same case writes. The number of rows comes from
   the binner's spec rather than from the local data, so every rank reports the
   same bins. ``residuals`` needs no reduction: a linear solve is itself
   collective, so its dictionary already holds the global residuals on every
   rank.

   The one exception is ``rows``: it writes the elements themselves rather than
   a reduction of them, and pybFoam binds no gather onto the master, so a
   decomposed run of a row table would silently drop every non-master rank's
   points. It raises a ``NotImplementedError`` naming the table instead — use an
   aggregator, or run that case serially.

Try it
------

:doc:`/auto_tutorials/example_05_postprocessing` walks through a laminar
lid-driven cavity and the two front doors: a declared table, a scripted one
built with ``|``, a per-table cadence, and a case-defined ``square`` node
registered in the script and used by name from the spec file. It plots the
cavity's kinetic energy, which rises from rest and plateaus as the primary
vortex reaches its steady strength.

The same case is in the repo at ``tutorials/postProcessing/cavity`` and runs
standalone:

.. code-block:: bash

    cd tutorials/postProcessing/cavity
    ./Allrun
    head -3 postProcessing/kinetic_energy.csv
    ./Allclean
