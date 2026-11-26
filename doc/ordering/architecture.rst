Ordering Framework Developer Guide
==================================

Overview
--------

The NeoFOAM ordering framework provides a common way to reorder computational
entities while keeping three concerns separate:

* determining an ordering;
* representing the resulting permutation;
* applying that permutation to CFD data.

This separation keeps ordering algorithms independent of ``NeoN::UnstructuredMesh``
and solver-specific code, while making execution and data movement explicit.

Current scope
-------------

The initial implementation targets local cell ordering of
``NeoN::UnstructuredMesh``.

The framework is designed to accommodate topology- or geometry-based strategies,
but distributed/global MPI ordering, dynamic or matrix-derived ordering,
external ordering libraries, and GPU-specific ordering implementations are
currently outside the scope.

Architecture
------------

An ordering operation follows this flow::

    CFD mesh/data
          |
          v
    ordering-domain input
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

**Ordering input**
    Adapts NeoN mesh data into the representation required by an algorithm.

**Ordering algorithm**
    Computes and returns a ``Permutation``. It does not modify the mesh,
    apply the result, or perform hidden data transfers.

**Permutation application**
    Applies the permutation to mesh- and field-associated data and remaps
    entity references so that the discretization remains consistent.

Core components
---------------

**Permutation**
    Represents the bijection between old and new entity indices. It has no
    mesh knowledge and no ordering policy.

**Ordering algorithm**
    Consumes ordering-domain information and produces a ``Permutation``.
    Different algorithms may require different inputs, for example entity
    count, cell adjacency, or cell coordinates.

**Ordering context / input provider**
    Forms the boundary between the CFD representation and the ordering domain.
    It exposes or derives only the information required by an algorithm rather
    than acting as a universal copy of the mesh.

**Permutation application**
    Reorders entity-associated data and remaps references independently of how
    the permutation was generated.

Execution and memory
--------------------

The design separates:

**Data residency**
    Where the data currently resides, such as host or device memory.

**Execution capability**
    Which executors an ordering implementation supports.

**Selected execution**
    The executor chosen for the current ordering operation.

The application or orchestration layer selects the execution environment.
The framework validates compatibility and reports unsupported combinations
explicitly.

Ordering and simulation execution are independent::

    import mesh
        |
        v
    prepare ordering data
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

Ordering implementations do not perform implicit host/device transfers.
Inputs are prepared explicitly in a compatible memory space so that data
movement and ordering cost remain visible.

NeoN integration
----------------

The framework uses NeoN's existing mesh, executor, and memory model.

Relevant mesh information is available through interfaces such as::

    mesh.nCells()
    mesh.faceOwners().view()
    mesh.faceNeighbors().view()
    mesh.cellCenters().view()
    mesh.exec()

NeoN remains the source of truth for mesh storage, execution, and memory
management.

Adding an ordering strategy
---------------------------

#. Identify the algorithm's required information.
#. Determine how that information is obtained from NeoN.
#. Expose or derive it through the ordering context/provider.
#. Implement the algorithm against the ordering-domain interface.
#. Return a valid ``Permutation`` without modifying the mesh.
#. Reuse the common permutation-application path.
#. Declare supported executors and relevant guarantees such as determinism.
#. Add unit tests independent of a complete CFD simulation and integration
   tests for the reordered mesh and associated data.

Application code should select strategies through the framework rather than
depending directly on concrete algorithm types.

Correctness and performance
---------------------------

A valid permutation must preserve the meaning of the discretization. For cell
ordering, cell-associated data, connectivity references, geometry, and fields
must remain correctly associated after reordering.

Tests should therefore cover both:

* the correctness of the generated ``Permutation``;
* the consistency of the resulting reordered CFD representation.

Performance should be evaluated as a complete preprocessing stage. Keep input
preparation, data transfers, ordering computation, permutation application,
and any transfer to the simulation executor separately measurable.

Current design direction
------------------------

The framework currently aims to:

* keep ``Permutation`` small and algorithm-independent;
* use the ordering context/provider as the mesh-to-algorithm boundary;
* keep algorithms independent of mesh ownership and data mutation;
* let orchestration control execution and compatibility;
* keep data movement explicit;
* use a common permutation-application path for consistency.

The main remaining lower-level decisions concern the concrete storage/API for
graph-based ordering inputs and the policy for geometry arrays affected by cell
reordering.
