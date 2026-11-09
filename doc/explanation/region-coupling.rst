Region coupling
===============

A multi-region case (``multiRun``) runs one solver per region, each with its own
graph and time loop on its own thread. Regions share nothing but a
**comm channel**, point to point; coupled ports exchange data over it.

.. list-table::
   :header-rows: 1

   * - part
     - owns
   * - coupled instance (``@coupled``: a boundary condition, a source)
     - what it sends and receives, a data model each, in the mesh order of its
       location; no patch, geometry or context.
   * - location (``Patch(name)``, ``Cells()``, ``Global(name)``)
     - the part of the mesh, and so the geometry: face centres and areas, cell
       centres and volumes, from the region's ``MeshGeometry``.
   * - solver
     - its coupling model, ``ctx.models["coupling"] = Participant(self.coupling,
       geometry, fields)``, and the binding of its instances,
       ``coupling(ctx).add(Patch("top"), bc)``: a **port**.
   * - case (``system/coupling.yaml``)
     - the **field ports** (``ports``), the **exchanges** (pairs of ports and
       their mapping) and the **coupling loops** (``couplings``).
   * - framework
     - one coupling op ``<region>.<loop>`` per loop of a region, placed in the
       region's graph; the loop's last region (its **leader**) decides
       convergence.
   * - ``multiRun``
     - reads ``participants`` and the case file, creates the comm channel, runs
       every region on its own thread (under ``mpirun``: on its own processes).

Regions that exchanges connect form a **group**; its last region in
``participants`` order agrees the time step. Each region reads its own files
under ``<region>/`` (``<region>/constant``, ``<region>/0``).

The coupled instance
--------------------

.. code-block:: python

    class WallTemperature(FaceData):     # one row per face of a Patch
        T: Array


    class WallHeatFlux(FaceData):
        q: Array


    @coupled
    class CoupledHeatFlux(NeoNPatch):
        @coupled.sends
        def wall_temperature(self) -> WallTemperature: ...

        @coupled.receives
        def heat_in(self, data: WallHeatFlux) -> None: ...   # already mapped


    # solver init
    coupling(ctx).add(Patch("top"), CoupledHeatFlux(...))

The annotations are the declaration (string annotations resolve, so
``from __future__ import annotations`` is fine). A class sends, receives or
both, each data field once; ``@coupled`` checks this as the class is defined.
The base class names the entity of a row: ``FaceData`` lives on a ``Patch``,
``CellData`` on ``Cells``, ``ValueData`` (``Scalar`` fields) and ``ProbeData``
(a row per probe point) on a ``Global``. Instances bound to one location merge
into its port. ``add`` raises ``ValueError`` when no exchange connects the port;
at start-up an array with another row count than its location raises.
``test/solver/multiRun/cases/dirichlet_neumann_slabs`` couples two slabs this
way (``test/solver/multiRun/slab_solver.py``).

**Field ports.** A region's field on a location needs no Python: the case
names it, and the solver's ``FieldAccess`` reads or writes it. ``direction``
is ``out`` (read, the default), ``in`` (write) or ``inout``; an exchange of two
field ports connects ``out`` to ``in`` or ``inout`` to ``inout``, and a field
on a location is written by one port (checked as the file loads).

.. code-block:: yaml

    ports:
      fluid:
        - {name: T_out, field: T, patch: outlet}             # Patch("outlet"), out
        - {name: T_cells, field: T}                          # the cells
        - {name: U_in, field: U, patch: inlet, direction: in}

The case file
-------------

.. code-block:: yaml

    exchanges:
      - {name: wall-T, ports: {fluid: wall, solid: right}}   # mapping: nearest-neighbor
      - {ports: {fluid: inlet, pipe: outlet}, mapping: identity}   # name: fluid-pipe
      - {name: mean, ports: {fluid: T_out, controller: T}, mapping: area-mean}
    couplings:                       # outer to inner; each exchange in exactly one
      - name: wall                   # the ops <region>.wall
        scheme: serial               # serial (default) | parallel
        maxIterations: 50            # default 100
        exchanges:
          - {name: wall-T, acceleration: {type: aitken, relaxation: 0.5}}
          - fluid-pipe               # a plain name: no acceleration
        convergence:                 # every measure must hold; data: a data field
          - {type: relative, data: T, limit: 1.0e-6}
          - {type: relative, data: q, limit: 1.0e-6}
        wraps:                       # per region; default: the whole time step
          fluid: {operations: [solve_T]}
          pipe: {operations: [inner_loop]}   # a loop, held whole
    commChannelTimeout: 600          # seconds; 0 or null: no timeout

``wraps`` places the loop's op around the named operations and everything
downstream of them; without it the op wraps the body of the loop before it, or
the whole time step. ``mapping``, ``acceleration`` and ``convergence`` name a
strategy of the families ``CouplingMapping``, ``CouplingAcceleration`` and
``CouplingConvergence`` by ``type``, with its options.

An exchange pairs what one port sends with what the other receives by the data
models' field names, one message per model; each direction needs a pair. The
mapping moves a model between the locations' entities:

.. list-table::
   :header-rows: 1

   * - mapping
     - maps
   * - ``nearest-neighbor`` (default), ``identity``
     - faces to faces, cells to cells (``identity``: also values)
   * - ``area-mean``, ``area-integral``
     - a patch's faces to one value, weighted by the face areas
   * - ``volume-mean``, ``volume-integral``
     - the cells to one value, weighted by the cell volumes
   * - ``uniform``
     - one value to every face or cell
   * - ``probe`` (``{type: probe, points: [[x, y, z], ...]}``)
     - the cells nearest each point to a row each (``ProbeData``)

The value mappings take scalar fields only. A region's data field (``T``,
``q``) is in at most one coupling loop.

**Serial or parallel.** Parallel: a port receives what its partner sent in the
iteration before. Serial: from a region that steps before its own, what it sent in
the same iteration, so a serial iteration waits for the partner's whole iteration. Hence a
serial coupling loop may neither enclose a loop coupling two of its regions
nor hold their time step: place it around named operations or make it
parallel. A steady region has no time step, so a serial loop may wrap it whole.

The file is checked as it loads, by the same ``check_exchanges`` as
``participant_exchanges``: unknown keys, types or options, an exchange in no loop
or in two, two exchanges or loops of one name, a port in two exchanges, a
``wraps`` region the loop does not couple, and a serial loop enclosing a loop
of two of its regions. A serial loop holding the time step is found as the
region's graph is resolved.

How it runs
-----------

.. code-block:: text

    multiRun.run()                          main thread
      CouplingConfig.participant_exchanges(...) -> ParticipantExchanges per region
      run_participants: one thread per region; the first error is raised
        |
        +-- thread "a":  instantiate(region="a", coupling=...).run()
        |     the coupling model adds a.coupling (placed by its wraps) and,
        |     transient, a.time_step between set_time_step and increment_time
        +-- thread "b" (leader): the same; its op holds the leader
                     ^
                     |   CommChannel: send / receive (source, target, tag)
                     v

**The time step.** Every region of a group sends the time step it asks for to
the leader, which sends back the smallest. Ops marked ``first`` (the time-step
ops, a solver's old-time storage) run in the first iteration of a coupling loop
only.

**An iteration.** An iteration of a coupling loop is a pass of its loop op, the
entry of ``ctx.loops`` it counts. The first loop test tells each partner what a
port sends and its geometry, checks the pairs, binds the mappings once and builds the leader. Each test then sends the ports' data, a stream per data model
(serial: to later partners at once), and the leader's decision, convergence
and acceleration (per exchange, over its streams) included, comes back; on
another iteration each port receives its partner's data, mapped and validated. All
regions leave the loop on the same iteration.

**The final iteration.** Once every measure holds, the leader runs one more iteration,
the final one; the last iteration ``maxIterations`` allows is final regardless (it
warns, ``CouplingNotConvergedWarning``). ``maxIterations: 1`` is explicit
coupling: one iteration, no warning. Ops marked ``final`` (such as
``write_output``) run in the final iteration only.

**Failures.** A region that fails closes the comm channel, so none waits for it
forever. A deadlock raises ``CouplingDeadlock`` at once; as a fallback a receive
raises ``CouplingTimeout`` after ``commChannelTimeout`` seconds.

**Observers.** Every region reports to the execution observer from its own
thread (``coupling_exchanged``, ``coupling_finished``, its graph), so hooks
must be thread safe; a hook that blocks pauses its region and those coupled
with it.

A NeoN solver: heatConductionNeoN
---------------------------------

``heatConductionNeoN`` solves ``dT/dt = laplacian(k/(rho cp), T) + Q/(rho cp)``
on NeoN (``constant/thermalProperties``). A patch of ``0/T`` is coupled when its
``coupled`` entry names the condition; its ``type`` is ``calculated``, so NeoN
leaves its values to the coupling:

.. code-block:: c++

    interface
    {
        type        calculated;
        coupled     coupledTemperature;   // Dirichlet side; or coupledHeatFlux
        value       uniform 350;
    }

``coupledTemperature`` takes the wall temperature and sends the heat flux
``q = k dT/dn`` into its region (``WallHeatFlux``); ``coupledHeatFlux`` takes
that flux as an explicit source in the cells of its faces and sends its wall
temperature (``WallTemperature``, both in ``neofoam.solver.regions``). The
solver binds each to ``Patch(name)``; ``NeoNGeometry``
(``neofoam.solver.neon_runtime``) gives the face centres, so the faces need not
match, and ``NeoNFields`` the field ports' access. Each region is an
OpenFOAM case of its own under ``<region>/``, beside the case's
``system/controlDict.yaml`` and ``system/coupling.yaml``.
``tutorials/heatConductionNeoN/coupledBlocks`` couples two blocks through 3
against 5 faces, serial with Aitken around ``solve_T``.

Regions on MPI processes
------------------------

Under ``mpirun`` each region runs on processes of its own instead of a thread
(``neofoam.solver.multiRun.mpi``; mpi4py, the extra ``neofoam[mpi]``). A region
entry gives its number of processes, a plain name is one::

    participants:
      left: {solver: mySolver, processes: 2}
      right: {solver: mySolver, processes: 2}

    mpirun -np 4 python -m neofoam.solver.multiRun

The regions take the ranks of ``MPI_COMM_WORLD`` in blocks, in the order of
``participants`` (``left`` ranks 0-1, ``right`` 2-3), and each process runs
only its region. Every region runs OpenFOAM with ``-parallel -world <region>``,
also a region on one process: OpenFOAM's world is the region's block of ranks,
and the region's case is decomposed into its processes, ``<region>/processor0``
.. ``processor<n-1>`` (``decomposePar -case <region>``; one part for one
process). OpenFOAM's ``-world`` split is collective over all processes, so
every region takes part in it. A NeoN region communicates on its world too:
``create_adapter_run_time`` makes OpenFOAM's world NeoN's default communicator
(``neon.set_default_communicator``, which also takes an mpi4py communicator
from Python, set before), so the region's NeoN ranks are its OpenFOAM ranks,
the numbers of its processor patches' neighbours, and the root of each region
logs. Before the run starts every process checks the case, and rank 0 says
what is wrong before the run aborts: the number of processes is not the sum of
the regions', a region is not decomposed into its processes, a region's solver
cannot run on part of the processes (a NeoN built without the default
communicator communicates on ``MPI_COMM_WORLD``), or the coupling or its
``commChannelTimeout`` is wrong. An error in a region, on any of its processes,
prints its traceback and aborts every process (``MPI_Abort``), so no region
waits for it. Without ``mpirun`` (or on one process) the regions run on
threads, and ``processes`` is not used
(``test/solver/multiRun/test_multiRun_mpi.py``).

**Coupled regions on processes.** Only a region's **root**, rank 0 of its
processes, talks to other regions. The roots exchange today's messages over an
``MpiTransport`` (``neofoam.framework.coupling.mpi``), on a ``Dup`` of
``MPI_COMM_WORLD``: every message is one pickle under one MPI tag, its string
tag inside, and a receive sorts what arrived into a mailbox per ``(source,
tag)``. MPI does not overtake between two processes, so messages arrive in the
order they were sent. A receive raises ``CouplingDeadlock`` at once when its
partner finished, ``CommChannelClosed`` when it failed, and ``CouplingTimeout``
after ``commChannelTimeout`` (``NEOFOAM_COUPLING_TIMEOUT``), naming what it waits
for and what arrived but was not received; a deadlock between processes is
told by that timeout only. A root that finishes tells the other roots and takes
their messages until each has finished too.

The other processes of a region follow their root over the region's
``RegionComm`` (``RegionCoupling.comm``). In each loop test of a coupling op
every process hands its part of each side to the root (its points once, then
what it sends each pass, in rank order: a process may have no face of the
patch). The root runs the exchange on the whole patch, as one process would:
mapping, controller, finiteness check and the observer (``coupling_exchanged``
and ``coupling_finished`` on the root only). Then it broadcasts the leader's
decision (another pass? the final one?) and scatters to each process its rows
of what each side receives; every process marks the final pass, so operations
marked ``final`` run on all of them. The time-step op agrees on the root and
broadcasts the time step. A plain (not ``Array``) field of a side is rank 0's
value, and the root raises if another process has another. A region on one
process has a ``RegionComm`` of size 1, which does nothing.

Gathering on the root is an intermediate step: it costs a gather and a scatter
per pass and holds the whole interface on the root. Distributed exchange and
mapping between the processes of the regions (bounding-box partners, reduced
residuals, the leader deciding on the root) are to replace it.
``coupledBlocks`` (``heatConductionNeoN``) runs on 1+1, 1+2 and 2+2 processes,
its interface split between a region's processes, as on threads.

Extension points
----------------

.. list-table::
   :header-rows: 1

   * - to change
     - extend
   * - mapping
     - a ``CouplingMapping``: ``kinds`` (the entities it maps) and
       ``bind(source, target)`` once on both ports' ``Geometry``, returning the
       ``data -> data`` map of every iteration.
   * - backend
     - a ``MeshGeometry`` (face centres and areas, cell centres and volumes)
       and a ``FieldAccess`` (read and write a field on a location).
   * - acceleration
     - a ``CouplingAcceleration``: ``__call__(carried data, iteration) -> data``.
   * - convergence
     - a ``CouplingConvergence``: ``__call__(list of StreamChange) -> bool``.
   * - comm channel
     - a ``CommChannel`` (a Protocol): ``send`` (never blocks), ``receive``
       (blocks; in order per tag), ``close``, ``start``, ``abort``.

A strategy is a pydantic model with a ``type: Literal[...]`` that registers
with its family as its module is imported; the case file then names it:

.. code-block:: python

    @CouplingMapping.register
    class RadialBasisMapping(CouplingMapping):
        type: Literal["rbf"] = "rbf"
        support_radius: float = Field(gt=0)

        def bind(self, source, target): ...
