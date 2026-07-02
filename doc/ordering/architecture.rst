Ordering Framework Architecture
===============================

.. contents::
   :depth: 2
   :local:

Overview
========

The Ordering Framework provides a reusable infrastructure for computing,
evaluating, storing, and applying permutations of mesh entities in NeoFOAM.

Its primary objective is to improve the performance of matrix assembly,
linear solvers, and other mesh-based computations by improving memory
locality and reducing graph bandwidth.

Unlike traditional implementations that focus on one particular ordering
algorithm, this framework is designed around stable abstractions.
Ordering algorithms are interchangeable components that consume mesh-derived
information and produce a common output:

::

    Permutation

The framework supports

- graph-based ordering
- geometric ordering
- hybrid ordering
- future GPU-aware ordering

without changing the surrounding infrastructure.

Motivation
==========

NeoFOAM converts OpenFOAM meshes into NeoN meshes before performing matrix
assembly and numerical solution.

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

The Ordering Framework is inserted after NeoN mesh construction.

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
=====

Functional goals
----------------

The framework shall

- compute permutations
- apply permutations
- cache permutations
- store permutations
- benchmark ordering algorithms
- compare multiple ordering strategies
- support external libraries

Non-functional goals
--------------------

The framework shall

- minimise memory allocations
- avoid unnecessary copies
- support CPU and GPU implementations
- be modular
- be extensible
- be thread-safe
- be easy to unit test

Non-goals
=========

The first version does **not** support

- distributed-memory ordering
- adaptive mesh refinement
- dynamic reordering during runtime
- GPU-native graph construction

These may be added in future versions.

Design Principles
=================

The Ordering Framework follows the following principles.

1. Design around stable abstractions.

2. Algorithms compute permutations only.

3. The framework owns orchestration.

4. Meshes are accessed through views.

5. Graphs are implementation details.

6. Expensive derived data is lazily computed.

7. Algorithms are stateless.

8. File I/O is separated from algorithms.

9. New algorithms should require minimal framework changes.

10. Performance optimisation must not reduce maintainability.

High-Level Architecture
=======================

::

                     NeoFOAM

                         │

             ┌───────────┴───────────┐

             ▼                       ▼

          NeoN Mesh             Benchmark

             │

             ▼

         MeshView

             │

             ▼

      OrderingContext

      ┌──────────────┬───────────────┐

      ▼              ▼               ▼

 GraphBuilder    Geometry      Future Providers

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

 ┌────┴───────────┐

 ▼                ▼

IO             Applicator

 │                │

 ▼                ▼

Cache      Reordered Mesh

Component Responsibilities
==========================

MeshView
--------

Provides read-only access to mesh information.

Responsibilities

- cell connectivity
- owner/neighbour
- cell centres
- boundary information

OrderingContext
---------------

Provides lazily computed mesh-derived information.

Responsibilities

- graph()
- cellCenters()
- boundingBox()
- partitionInfo()

CSRGraph
--------

Stores graph connectivity.

GraphBuilder
------------

Constructs CSRGraph from MeshView.

Permutation
-----------

Represents

::

    old index → new index

Responsibilities

- validate()
- inverse()
- compose()

PermutationApplicator
---------------------

Applies permutations to

- mesh
- fields
- matrices

OrderingAlgorithm
-----------------

Consumes

::

    OrderingContext

Produces

::

    Permutation

Algorithms never modify meshes.

OrderingRegistry
----------------

Maps algorithm names to implementations.

PermutationIO
-------------

Reads and writes permutation files.

PermutationCache
----------------

Stores cached permutations for later reuse.

Module Dependencies
===================

Dependencies always point downward.

::

    Applications

          │

          ▼

      Solver

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

Ordering must never depend on Solver.

Graph must never depend on Ordering.

Extension Points
================

Adding an ordering algorithm
----------------------------

Implement

::

    compute(const OrderingContext&)

Register the algorithm.

Add unit tests.

No framework modifications should be necessary.

Adding a graph representation
-----------------------------

Implement the Graph concept.

GraphBuilder may construct it.

Existing algorithms remain unchanged.

Adding a benchmark
------------------

Implement a BenchmarkRunner.

Benchmarks are independent from algorithms.

Persistence
===========

Permutations are persistent objects.

The framework stores

- permutation
- algorithm
- parameters
- mesh fingerprint
- NeoFOAM version

The framework does **not** store reordered meshes.

Workflow
========

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

      ├───────────────┐

      │               │

      ▼               ▼

Load            Compute

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
