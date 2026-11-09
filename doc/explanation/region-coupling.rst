Region coupling
===============

A multi-region case (``multiRun``) runs one solver per region. Every region is
an independent solver: it builds its own graph, runs its own time loop on its
own thread and shares nothing with the other regions but a **comm_channel**, a
point-to-point channel. Regions exchange data over *exchanges*. Who does what:

.. list-table::
   :header-rows: 1

   * - part
     - owns
   * - boundary condition (``@coupled``)
     - its side: the field it couples, what it sends and receives (one pydantic
       model per direction), its points. It adds itself to its region's
       **coupling model** (``Coupling.of(ctx).add(self)``).
   * - case (``system/regionCoupling.yaml``)
     - which patches are paired (``exchanges``), the **mapping** of each pair, and
       the coupling nodes (``couplings``): their exchanges, scheme,
       acceleration, convergence, ``maxIterations`` and placement.
   * - solver
     - if it supports coupling, its **coupling model** (``Coupling``,
       ``ctx.models["coupling"] = Coupling(self.coupling)``), added in its init
       before its boundary conditions.
   * - framework
     - adds the coupling model's operations to the region's graph: one
       **coupling op** (``CouplingOp``, ``<region>.<node>``) per **coupling
       node** (``CouplingNode``) of the region, placed by the resolver as the
       node says. The **leader** of a node owns its **controller**
       (``CouplingController``).
   * - ``multiRun``
     - a launcher: reads ``regionSolvers`` and the case file, creates the
       comm_channel, runs every region on its own thread.

Vocabulary: a **region** is one solver; an **exchange** is one line between two
patches of two regions; the **mapping** (a ``Mapping``, by its ``type``) maps point
data of the partner onto a side's own points; a region's **coupling op** is a
loop around its solving operations whose test exchanges its sides' data; regions
connected by exchanges form a **group**, whose last region in ``regionSolvers``
order leads it in agreeing the time step; a **coupling node** is a named coupling loop over
some exchanges (its scheme, convergence, ``maxIterations``, an acceleration per
exchange, its placement per region; nodes are listed outer to inner), whose last region in
stepping order is the node's **leader**: its controller decides when the node's
data converged; a side is keyed by ``(region, patch, field)``. ``multiRun``
builds the nodes the case file lists under ``couplings``; without them it puts
every exchange in one node, ``coupling``.

The coupled boundary condition
------------------------------

.. code-block:: python

    from pydantic import BaseModel

    from neofoam.framework.coupling import Array, coupled


    class WallTemperature(BaseModel):
        T: Array             # one value per point: mapped


    class WallHeatFlux(BaseModel):
        q: Array
        h: float             # one value for the interface: handed over as it is


    @coupled                 # declares only: what it sends and receives, its points
    class CoupledHeatFlux(Patch):        # sets self.field and self.patch
        def __init__(self, ctx, field, patch):
            super().__init__(ctx, field, patch)
            Coupling.of(ctx).add(self)   # a side of its region's coupling op

        @coupled.coordinates
        def points(self) -> Array: ...

        @coupled.sends
        def send(self) -> WallTemperature: ...

        @coupled.receives
        def receive(self, data: WallHeatFlux) -> None: ...   # already mapped

The annotations are the declaration. One model per direction; the same model
both ways is the special case of a symmetric exchange. ``@coupled`` only
declares (``__coupled__`` on the class). The instance adds itself to its
region's coupling model: ``Coupling.of(ctx)`` raises ``ValueError`` when the
region's solver added none (it does not support coupling), ``add`` when the
instance sets no ``field`` or ``patch`` or no exchange connects its patch (also
when the case does not couple the region). Any ``@coupled`` model, not only a boundary
condition, couples this way. It names no partner, no loop and no mapping. The example
``test/solver/multiRun/cases/dirichlet_neumann_slabs`` couples two slabs
Dirichlet-Neumann with ``coupledTemperature`` and ``coupledHeatFlux``
(``test/solver/multiRun/slab_solver.py``).

The case file
-------------

.. code-block:: yaml

    exchanges:
      - sides: {fluid: wall, solid: right}
        mapping: nearest-neighbor    # or identity (default nearest-neighbor)
        name: cht                    # optional, default "<region>-<region>"
      - sides: {solid: left, air: top}
    scheme: parallel                 # parallel (Jacobi, the default) | serial (Gauss-Seidel)
    acceleration: {type: aitken, relaxation: 0.5}
    convergence:
      - {type: relative, data: T, limit: 1.0e-6}
    maxIterations: 100

Without ``couplings`` the file describes one node, ``coupling``, holding every
exchange: ``acceleration`` (of every exchange, default none), ``convergence``
(required) and ``maxIterations`` (default 100) are its own, and its op wraps the
whole time step of each region. ``couplings`` places the nodes instead:

.. code-block:: yaml

    exchanges:
      - {name: wall-T, sides: {fluid: wall, solid: right}}
      - {name: wall-p, sides: {fluid: inlet, pipe: outlet}}
    couplings:                       # outer to inner
      - name: wall                   # unique; the ops <region>.wall
        scheme: parallel             # default: the top-level scheme (default parallel)
        maxIterations: 50            # default 100
        exchanges:                   # each exchange in exactly one node
          - {name: wall-T, acceleration: {type: aitken, relaxation: 0.5}}
          - wall-p                   # a plain name: no acceleration
        convergence:                 # every measure must hold
          - {type: relative, data: T, limit: 1.0e-6}
        wraps:                       # per region of the node; default: the whole time step
          fluid: {operations: [solve_T], downstream: true}

The nodes are listed outer to inner: a node's op without ``operations`` wraps the
body of the op of the node before it (or the whole time step). ``wraps`` follows
``Placement``: with ``operations`` it wraps those operations and, if
``downstream`` (the default), every operation downstream of them; an operation
name the region's graph lacks is reported when the graph is built. With
``couplings`` the top-level ``acceleration``, ``convergence`` and
``maxIterations`` are an error; the top-level ``scheme`` is the scheme of every
node that names none.

**Serial or parallel**, per node. Parallel (the default): a side receives what
its partner sent in the pass before. Serial: a side receives from a region that
steps before its own what it sent in the same pass, so a serial pass of a region
waits for its partner's whole pass. Two rules follow, both checked before
anything runs (``ValueError``):

- a serial node encloses no node (one listed after it) that couples two of its
  regions as well: the partner's pass needs the inner node of both regions, so
  neither comes back. A chain (the inner node shares one region) is fine
  (``region_couplings``);
- a serial node does not hold the time step of its regions (the time-step op
  waits for the leader, which waits for the partner's pass): place it around
  named operations (``wraps.<region>.operations``) or make it parallel
  (``Coupling.check``, as the region's graph is resolved). A steady region has
  no time step, so a serial node may wrap it whole.

``mapping``, ``acceleration`` and ``convergence[]`` each name a strategy by its
``type``, with its options (``mapping`` may be the type alone): a member of the
plugin families ``Mapping``, ``Acceleration`` and ``Convergence`` (see
`Extension points`_). The file is checked against them on load: an unknown
type lists the registered ones, an unknown or missing option the strategy's
options; as against unknown keys, two nodes of one name, an
exchange in no node or in two, and a ``wraps`` region the node does not couple.
A patch is in at most one exchange and carries one coupled side.

Architecture
------------

.. code-block:: text

    multiRun.run()                          main thread
      reads regionSolvers + regionCoupling.yaml
      region_couplings(...) -> RegionCoupling per coupled region
      run_regions: one thread per region, joined; the first error is raised
        |
        +-- thread "a":  instantiate(region="a", coupling=...)
        |     initialize()       the solver: ctx.models["coupling"] = Coupling(self.coupling),
        |                        then its BCs; a @coupled BC adds itself
        |     execution_graph    model ops += ctx.models["coupling"].operations(),
        |                        the time-step op a.time_step (transient)
        |     resolve_operations DAGResolver places a.coupling around the time step;
        |                        Coupling.check: no time step in a serial node
        |     run()              time_loop { a.coupling { set_time_step, a.time_step,
        |                                                 increment_time, solve_x } }
        |
        +-- thread "b" (leader): the same, its coupling op owns the controller
                     ^
                     |   CommChannel: send / receive (source, target, tag)
                     v

**Where the op goes.** A model's loop declares where it goes
(``OperationMetadata.wraps``, a ``Placement``). Without operation names it wraps
the whole body of the time loop; with them the resolver moves those operations,
and every operation downstream of them (over ``depends_on`` / ``before``), into
the loop, which takes the place of the last of them. The operations must lie in
one scope; else it raises ``ValueError``.

**The coupling model** builds its operations once, after init: one
``CouplingOp`` per node of the region, outer to inner, from the sides of the
node's exchanges (``wraps=`` the node's placement for the region, by default
``Placement()``); none when the case does not couple the region. It raises
``ValueError`` when an exchange of the region has no side (also when the region
has none at all), a patch two sides, or one field has sides in two nodes.

**The time step.** In a transient coupled region (``ctx.time`` is set) the
framework adds the time-step op ``<region>.time_step`` (``Coupling.time_step``)
to the region's graph, after the solutionLoop's ``set_time_step`` and before its
``increment_time``; a region without them raises ``ValueError``. Every region of
the group sends the time step it asks for (``ctx.time.delta_t``) to the leader,
which sends back the smallest (tag ``time/dt``); each region writes it into its
``ctx.time.delta_t``, which ``increment_time`` advances by. So the coupled
regions step the smallest time step any of them asks for. The op is marked
``first`` (``OperationMetadata.first``), as are ``set_time_step`` and
``increment_time``: inside a coupling loop (the default placement around the
whole time step) they run in its first pass only, once per time step; an op
marked ``first`` runs while every wrapping loop around it is in its first pass
(``ctx.loops``). A solver's own start of a time step (storing the old time
level) is marked ``first`` the same way.

**The final pass.** Once every convergence measure of a node holds, its leader
runs exactly one more pass, the **final** one, and stops after it without
checking again; the last pass ``maxIterations`` allows is final in advance (no
pass more; it warns when the data did not converge), and with ``maxIterations``
1 the only pass is first and final. The acceleration relaxes the carried data
of the final pass as of any other. The leader sends the flag with its decision,
so every region of the node runs the same passes; each coupling op marks the
pass on its loop entry (``LoopEntry.final``, ``Context.mark_final``). An
operation marked ``final`` (``OperationMetadata.final``, the ``final=`` of the
``operation`` decorators), such as the fieldWriter's ``write_output``, runs
while every wrapping loop around it is in its final pass: once per time step,
on the converged data. Outside wrapping loops it runs every time. The resolver
raises when an op that runs in the other passes of a wrapping loop depends on a
``final`` op in it, and an op cannot be ``first`` and ``final``. For debugging,
``ctx.run_every_pass`` (operation names, per region) runs the ``final`` ops it
names in every pass; naming an op that is not marked ``final`` raises.

A **steady** solver (outer iterations, no time step) gets no time-step op; the
coupling needs no solution loop (``test/solver/multiRun/steady_solver.py``).

As a time step's coupling starts, the leader compares the regions' time
(``ctx.time.value``) and raises when they differ; regions without a time
(``ctx.time`` is ``None``, steady) compare equal, so the check passes.

**What a region knows.** Its ``RegionCoupling`` (``Coupling.coupling``): its name,
its group (in stepping order), the group's exchanges (``Link``: name, two ends
``(region, patch)``, its mapping), its nodes (each with its scheme), the controller
factory (``node_controller`` by default) and the comm channel. No region reads another region's objects.

**A run of a coupling op.** Its first loop test sends every partner what its
side sends and receives (the models' field kinds) and its points, and the leader
which sides it has; each region then checks that its side receives what the
partner sends and binds its mapping to the partner's points and its own, once
(which fails when it cannot map between them; every pass then maps with the
bound mapping), and the leader builds its controller. Each loop test then:

1. computes what every side of the region sends now;
2. after a pass, serial: sends it to every partner that steps later, which
   receives it in the same pass;
3. sends it to the leader and waits for its decision. The leader checks the
   regions started the time step at the same time (if they have one), runs convergence and
   acceleration (on what the partners receive as carried), and sends every region
   to go on (with the carried data its sides receive and whether the pass is the
   final one) or to stop;
4. on another pass, hands every side what its partner sent (fresh from a region
   that stepped before its own, serial; else carried by the leader), mapped onto
   its points, validated as its ``receives`` model and reported to the observer.

All regions leave the loop on the same pass. Receives block; a region that fails
closes the comm channel, so no region waits for it forever. A deadlock (every running
region waits for a message none sent, or one waits for a region that finished)
raises ``CouplingDeadlock`` at once; as a fallback a receive raises
``CouplingTimeout`` after ``transportTimeout`` seconds (top-level key of
``system/regionCoupling.yaml``, default 600; ``0`` or ``null`` waits without end),
which the environment variable ``NEOFOAM_COUPLING_TIMEOUT`` overrides. Both list
who waits for what and the messages sent but not received.

**Read API** of a ``CouplingOp``: ``region``, ``label``, ``sides``
(``ExchangeSide``: ``region``, ``patch``, ``field``, ``end`` ``(region, patch)``,
``sends``, ``receives``, ``mapped``, ``points``), ``exchange_of(side)`` (``Exchange``:
``name``, ``partner`` ``(region, patch)``, ``mapping``), ``partner(side)``
(``"<region>.<patch>"``), ``node``, ``leader``, ``is_leader`` and ``controller`` (the leader's, built at
its first loop test: ``label``, ``parallel``, ``sides`` (by end), ``iteration``,
``final``, ``outcomes``, ``residuals``).

**Report.** At the end of each time step the leader's controller records a
``CouplingOutcome`` and calls the observer's ``coupling_finished(controller,
outcome)``; it warns (``CouplingNotConvergedWarning``) when ``maxIterations``
stopped it. Each coupling op calls ``coupling_exchanged(coupling, side, data)``
from its loop test with the data it hands over. Every region reports to the
observer from its own thread; its graph with ``operations(operations, ctx)``,
``ctx.region`` naming it.

Extension points
----------------

.. list-table::
   :header-rows: 1

   * - to change
     - extend
   * - Mapping
     - a ``Mapping`` subclass: ``bind(source, target)`` on the partner's and
       the side's points, once, returns the ``BoundMapping`` ``data -> data``
       of every pass (yaml ``mapping:``).
   * - Acceleration
     - an ``Acceleration`` subclass: ``__call__(carried Array data, iteration)
       -> data``; it keeps its last passes in private attributes (yaml
       ``acceleration:``).
   * - Convergence
     - a ``Convergence`` subclass: ``__call__(list of SideChange) -> bool``
       (yaml ``convergence[]``).
   * - CommChannel
     - a ``CommChannel``: ``send(source, target, tag, message)`` (never blocks),
       ``receive(source, target, tag)`` (blocks; in order per tag) and
       ``close(region)``; messages are plain data. ``InProcessCommChannel`` connects
       threads of one process.

The three strategy families are families of NeoFOAM's ``PluginSystem``, as the
viscosity models or the post-processing writers are: each strategy is a
pydantic model with a ``type: Literal[...]`` that selects it, its options as
its fields (an unknown one is refused), registered with its family
(``@Mapping.register``). The built-in ones are members as well: the mappings
``identity`` and ``nearest-neighbor``, the accelerations ``constant`` and
``aitken`` (option ``relaxation``) and the convergence ``relative`` (options
``data``, ``limit``). ``strategy(family, options)`` builds one from its yaml
options.

A package of its own, e.g. a ``neofoam-precice`` with preCICE's RBF mapping and
IQN-ILS acceleration, adds its strategies without NeoFOAM importing it: an
entry point of the group ``neofoam.coupling`` names a module that registers
them as it is imported.

.. code-block:: toml

    [project.entry-points."neofoam.coupling"]
    precice = "neofoam_precice.coupling"

.. code-block:: python

    # neofoam_precice/coupling.py
    @Mapping.register
    class RadialBasisMapping(Mapping):
        type: Literal["rbf"] = "rbf"
        support_radius: float = Field(gt=0)

        def bind(self, source, target): ...

    @Acceleration.register
    class IQNILSAcceleration(Acceleration):
        type: Literal["iqn-ils"] = "iqn-ils"
        relaxation: float = Field(gt=0, le=1)
        columns: int = 10

        def __call__(self, sent, iteration): ...

.. code-block:: yaml

    exchanges:
      - sides: {fluid: wall, solid: right}
        mapping: {type: rbf, support_radius: 0.01}
    acceleration: {type: iqn-ils, relaxation: 0.5, columns: 20}

``multiRun`` imports every plugin once, before ``system/regionCoupling.yaml``
is read and before any region starts (``load_plugins``; safe to call from
several threads, the first call imports, the others wait), so no region sees
the families change. A plugin that fails to import, or registers a type that
is registered already, raises an ``ImportError`` naming it. Nothing is
overridden.

What is exchanged follows from the side's ``sends`` / ``receives`` models:
``Array`` fields are mapped, relaxed and compared, plain fields handed over as
they are.
