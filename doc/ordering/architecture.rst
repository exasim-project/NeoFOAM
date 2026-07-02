Ordering Framework Architecture
===============================

.. contents::
   :depth: 3
   :local:

Overview
--------

The Ordering Framework provides reusable infrastructure for computing,
evaluating, storing, and applying permutations of mesh entities in NeoFOAM.

Its primary objective is to improve the performance of matrix assembly,
linear solvers, and other mesh-based computations by improving memory
locality and reducing graph bandwidth.

Rather than being built around a single ordering algorithm, the framework is
designed around stable abstractions. Ordering algorithms are interchangeable
components that consume mesh-derived information and produce a common output:

::

   Permutation

The framework supports

- graph-based ordering
- geometric ordering
- hybrid ordering
- future GPU-aware ordering

without requiring changes to the surrounding infrastructure.

Motivation
----------

NeoFOAM currently converts an OpenFOAM mesh into a NeoN mesh before matrix
assembly and solution.

::

   OpenFOAM Mesh
         │
         ▼
   Mesh Adapter
         │
         ▼
     NeoN Mesh
         │
         ▼
   Matrix Assembly
         │
         ▼
   Linear Solver

The Ordering Framework is inserted immediately after NeoN mesh construction.

::

   OpenFOAM Mesh
         │
         ▼
     NeoN Mesh
         │
         ▼
      Ordering
         │
         ▼
   Reordered NeoN Mesh
         │
         ▼
   Matrix Assembly
         │
         ▼
   Linear Solver

The reordered mesh is then used throughout the simulation.

Goals
-----

Functional goals
~~~~~~~~~~~~~~~~

The framework shall

- compute permutations
- apply permutations
- cache permutations
- store permutations
- benchmark ordering algorithms
- compare ordering strategies
- support external libraries

Non-functional goals
~~~~~~~~~~~~~~~~~~~~

The framework shall

- minimise memory allocations
- avoid unnecessary copies
- support CPU and GPU implementations
- be modular
- be extensible
- be thread-safe
- be easy to unit test

Non-goals
---------

The first version does **not** support

- distributed-memory ordering
- adaptive mesh refinement
- dynamic runtime reordering
- GPU-native graph construction

These capabilities may be added in future versions.

Design Principles
-----------------

The framework follows these principles.

1. Design around stable abstractions.
2. Algorithms compute permutations only.
3. The framework owns orchestration.
4. Meshes are accessed through views.
5. Graphs are implementation details.
6. Expensive derived data is computed lazily.
7. Algorithms are stateless.
8. File I/O is separated from algorithms.
9. New algorithms require minimal framework changes.
10. Performance optimisations must not reduce maintainability.

High-Level Architecture
-----------------------

::

                           NeoFOAM
                               │
              ┌────────────────┴────────────────┐
              │                                 │
              ▼                                 ▼
         NeoN Mesh                       Benchmarking
              │
              ▼
          MeshView
              │
              ▼
      OrderingContext
              │
      ┌───────┼───────────┐
      │       │           │
      ▼       ▼           ▼
 GraphBuilder Geometry  Future Providers
      │
      ▼
   CSRGraph
      │
      ▼
 OrderingAlgorithm
      │
      ▼
  Permutation
      │
  ┌───┴──────────────┐
  │                  │
  ▼                  ▼
PermutationIO   PermutationApplicator
  │                  │
  ▼                  ▼
 Cache         Reordered NeoN Mesh

Component Responsibilities
--------------------------

MeshView
~~~~~~~~

Provides read-only access to mesh information.

Responsibilities

- cell connectivity
- owner/neighbour relationships
- cell centres
- boundary information

OrderingContext
~~~~~~~~~~~~~~~

Provides lazily computed mesh-derived information.

Responsibilities

- graph()
- cellCenters()
- boundingBox()
- partitionInfo()

CSRGraph
~~~~~~~~

Stores graph connectivity in compressed sparse row (CSR) format.

GraphBuilder
~~~~~~~~~~~~

Constructs a CSRGraph from a MeshView.

Permutation
~~~~~~~~~~~

Represents

::

   old index → new index

Responsibilities

- validate()
- inverse()
- compose()

PermutationApplicator
~~~~~~~~~~~~~~~~~~~~~

Applies permutations to

- meshes
- fields
- matrices

OrderingAlgorithm
~~~~~~~~~~~~~~~~~

Consumes

::

   OrderingContext

Produces

::

   Permutation

Algorithms never modify mesh data.

OrderingRegistry
~~~~~~~~~~~~~~~~

Maps algorithm names to implementations.

PermutationIO
~~~~~~~~~~~~~

Reads and writes permutation files.

PermutationCache
~~~~~~~~~~~~~~~~

Stores cached permutations for later reuse.

Module Dependencies
-------------------

Dependencies always point downward.

::

   Applications
         │
         ▼
      Solvers
         │
         ▼
      Ordering
         │
         ▼
        Graph
         │
         ▼
         Mesh
         │
         ▼
         Core

Ordering must never depend on solver modules.

Graph must never depend on ordering modules.

Extension Points
----------------

Adding an ordering algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implement

::

   compute(const OrderingContext&)

Register the algorithm.

Add unit tests.

No framework modifications should be necessary.

Adding a graph representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implement the Graph concept.

Extend GraphBuilder to construct the new graph type.

Existing algorithms should remain unchanged.

Adding a benchmark
~~~~~~~~~~~~~~~~~~

Implement a BenchmarkRunner.

Benchmarks are independent from ordering algorithms.

Persistence
-----------

Permutations are persistent objects.

The framework stores

- permutation
- algorithm
- parameters
- mesh fingerprint
- NeoFOAM version

The framework does **not** store reordered meshes.

Workflow
--------

The framework executes the following workflow.

::

      Mesh
       │
       ▼
MeshFingerprint
       │
       ▼
  Cache Lookup
       │
   ┌───┴────┐
   │        │
   ▼        ▼
 Load    Compute
            │
            ▼
      Permutation
            │
            ▼
        Validate
            │
            ▼
    Apply Ordering
            │
            ▼
    Matrix Assembly