Ordering Framework Developer Guide
==================================

Overview
--------

The NeoFOAM ordering framework provides a common infrastructure for changing the
ordering of computational entities in a CFD discretization while keeping the
ordering algorithm itself independent from the mesh implementation and from the
code that applies the resulting permutation.

The current design is intentionally layered. An ordering algorithm is treated as
an independent computational component: it receives the information it needs,
computes a permutation, and returns that permutation. A separate part of the
framework is responsible for interpreting the permutation and applying it to the
mesh and associated data.

This separation is the central design principle of the framework. It allows new
ordering strategies to be added without coupling them to NeoFOAM solver logic,
and it keeps data movement, execution selection, and permutation application
visible at the framework level.

Current scope
-------------

The initial implementation targets local cell ordering for ``NeoN::UnstructuredMesh``.
The framework is designed so that ordering strategies can use different kinds of
information, including connectivity or geometry, but the first implementation
focuses on establishing the common infrastructure rather than supporting every
possible ordering algorithm.

The current architecture is deliberately not a distributed ordering framework.
MPI/global ordering, dynamic reordering, matrix-derived ordering, external
ordering libraries, and GPU-specific ordering algorithms are outside the present
scope. The interfaces should nevertheless avoid assumptions that would prevent
these capabilities from being introduced later.

Architectural model
-------------------

At a high level, an ordering operation follows this path::

    CFD mesh/data
        |
        v
    ordering input
        |
        v
    ordering algorithm
        |
        v
    Permutation
        |
        v
    permutation application
        |
        v
    reordered CFD data

The important architectural distinction is that these stages have different
responsibilities.

**Input preparation** adapts the CFD representation to the information required
by an ordering method. For example, a graph-based method should consume a cell
adjacency representation rather than direct knowledge of ``faceOwners`` or
``faceNeighbors``.

**Ordering algorithms** operate only on ordering-domain data and produce a
``Permutation``. They do not own or modify the source mesh, apply the result, or
hide data transfers.

**Permutation application** translates the abstract permutation into concrete
changes to mesh-associated data. It is responsible for preserving consistency
between reordered entity data and references such as connectivity indices.

This separation keeps algorithm development, mesh adaptation, and data mutation
independent enough to test and evolve them separately.

Core components
---------------

The framework is organized around four main concepts.

Permutation
~~~~~~~~~~~

``Permutation`` is the common representation of an ordering result. It represents
the bijection between the original and reordered entity indices and provides the
mapping operations needed by later stages.

It deliberately contains no mesh knowledge and no ordering policy. In particular,
it does not decide how a permutation is generated or how it should be applied to
CFD data.

Ordering algorithm
~~~~~~~~~~~~~~~~~~

An ordering algorithm computes a permutation from ordering-domain information.
The algorithm is therefore independent of the concrete ``NeoN::UnstructuredMesh``
representation.

Examples of the intended separation are:

* identity ordering needs only the number of cells;
* a graph-based method such as RCM needs cell adjacency;
* a geometry-based method such as Morton ordering needs cell coordinates.

The algorithm interface should expose these requirements without forcing every
algorithm to depend on a large, universal mesh-data object.

Ordering context / input provider
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ordering context forms the boundary between the CFD mesh representation and
the ordering domain. Its purpose is to expose or derive the information required
by an algorithm while hiding unrelated mesh internals.

The context should therefore be understood as an adapter/provider rather than a
complete copy of the mesh. This keeps algorithm dependencies small and makes it
possible to prepare only the information that a particular ordering strategy
actually needs.

Permutation application
~~~~~~~~~~~~~~~~~~~~~~~

Permutation application is a separate operation because computing an ordering and
mutating a CFD discretization are fundamentally different tasks.

For cell ordering, application may involve both reordering cell-associated arrays
and remapping references that point to cells. Connectivity, geometry data, and
field data must remain mutually consistent after the transformation.

The application layer must not depend on how the permutation was produced. The
same application mechanism should therefore be reusable for identity ordering,
graph ordering, geometry ordering, and future strategies.

Execution model
---------------

A key part of the current design is the separation of three concepts:

**Data residency** describes where the relevant data currently lives, for example
host or device memory.

**Execution capability** describes where a particular ordering implementation can
run. A serial implementation may require a ``SerialExecutor``, while a parallel
implementation may require a CPU or GPU executor.

**Selected execution** describes the executor chosen for the current ordering
operation.

These concepts are intentionally not collapsed into the ordering algorithm. The
application or ordering orchestration layer chooses the execution environment,
and the framework verifies that the selected environment is supported by the
chosen implementation.

The design therefore treats an unsupported implementation/executor combination
as an explicit compatibility failure rather than silently changing the selected
algorithm or execution mode.

Ordering execution and simulation execution are also distinct. The executor used
to compute and apply the ordering does not necessarily have to be the executor
used by the subsequent CFD simulation.

A typical flow is::

    import mesh on host
          |
          v
    prepare data for ordering
          |
          v
    compute permutation
          |
          v
    apply permutation
          |
          +----> transfer if simulation executor differs
          |
          v
    run simulation

This makes data movement explicit and keeps ordering costs measurable.

Memory and data movement
------------------------

Ordering implementations are not responsible for implicit host/device transfers.
The framework prepares their inputs in a memory space compatible with the selected
execution environment before the algorithm is invoked.

This rule is important for two reasons. First, it prevents hidden transfers from
appearing inside otherwise simple ordering algorithms. Second, it allows the
framework to measure input preparation, data movement, ordering computation, and
permutation application as separate stages.

The current design already aligns with the executor-aware memory model in NeoN:
read-only views can be used to access mesh data, and explicit transfer mechanisms
can prepare data for another execution environment.

For the current serial ordering scope, this means that a graph or coordinate input
may be materialized on the host before the algorithm runs. The architecture does
not require GPU ordering today, but it avoids introducing host-only assumptions
into the algorithm interface.

Interaction with NeoN
---------------------

The framework is integrated with ``NeoN`` rather than introducing a second mesh or
executor model.

The current design relies on NeoN facilities such as::

    mesh.nCells()
    mesh.faceOwners().view()
    mesh.faceNeighbors().view()
    mesh.cellCenters().view()
    mesh.exec()

These interfaces provide the source information needed to construct ordering-domain
inputs while preserving the existing ownership and execution model.

The framework should continue to treat NeoN as the source of truth for mesh storage,
execution, and memory management. Ordering-specific abstractions should exist only
where they provide a useful boundary for algorithms or permutation application.

Typical developer workflow
--------------------------

A developer adding a new ordering strategy should think in terms of the following
workflow rather than modifying the solver directly.

#. Define the mathematical/data requirements of the ordering algorithm.
#. Identify which of those requirements can be obtained from NeoN directly and
   which need an adapter or derived representation.
#. Expose those requirements through the ordering context/provider.
#. Implement the ordering algorithm against the ordering-domain interface.
#. Return a valid ``Permutation`` without modifying the source mesh.
#. Reuse the common permutation-application path to update mesh-associated data.
#. Declare the execution environments supported by the implementation.
#. Add unit tests for the algorithm independently of a complete CFD simulation.
#. Add integration coverage for the resulting reordered mesh and associated data.

This workflow is intended to keep algorithm-specific code small and make the
framework responsible for orchestration, validation, and data consistency.

Adding a new ordering algorithm
--------------------------------

A new implementation should not require changes to existing ordering algorithms.
In practice, the implementation should answer four questions:

* What information does the algorithm require?
* How is that information represented in the ordering domain?
* Which executors can run the implementation?
* What guarantees does the implementation provide, such as determinism?

The algorithm should then be registered or selected through the framework's
application-level interface rather than being instantiated directly by solver code.
This keeps application code independent of concrete ordering implementation types.

Correctness considerations
---------------------------

The most important correctness property is not merely that a permutation is
bijective. The reordered CFD representation must preserve the semantics of the
original discretization.

For cell ordering, developers should consider at least:

* cell-associated values are moved to their new positions;
* cell references in connectivity are remapped consistently;
* geometry remains associated with the correct cells;
* fields remain associated with the same physical entities;
* the resulting mesh can still be consumed by the solver without changing the
  numerical meaning of the discretization.

Because permutation application is separated from permutation generation, these
properties should be tested at both levels: algorithm tests verify the permutation,
while integration tests verify the reordered CFD representation.

Performance considerations
---------------------------

The framework is designed to keep the main performance costs visible. At minimum,
developers should be able to distinguish:

* preparation or construction of ordering-domain data;
* memory transfers required for preparation;
* ordering computation;
* permutation application;
* transfer to the simulation execution environment, when required.

This separation is especially important when comparing ordering strategies. A
faster ordering kernel is not necessarily a faster overall preprocessing stage if
its input construction or data movement dominates the runtime.

Current design direction
------------------------

The current implementation direction can be summarized as follows:

* keep ``Permutation`` as a small, algorithm-independent value abstraction;
* use an ordering context/provider to bridge NeoN mesh data and algorithm inputs;
* keep algorithms unaware of concrete mesh ownership and permutation application;
* let the orchestration layer choose execution and validate compatibility;
* keep data movement explicit;
* use a common permutation-application path for mesh and field consistency;
* design interfaces so additional CPU/GPU or more specialized ordering strategies
  can be added without redesigning the existing solver workflow.

This gives the framework a stable separation of concerns while leaving implementation
choices open where the design is not yet finalized, such as the concrete storage and
API of graph-based ordering inputs.

Implementation status and open design points
--------------------------------------------

The architectural boundaries are established, while some lower-level implementation
details remain intentionally open. In particular, the exact graph-storage abstraction
for connectivity-based ordering is not yet fixed, and policies for which geometry
arrays must be reordered when cell ordering changes require explicit definition.

These decisions should be made at the framework level rather than embedded in an
individual ordering implementation, because they affect interoperability between
algorithms and the common permutation-application stage.

See also
--------

* :doc:`architecture`
* :doc:`../requirements`
