Model discovery: plugin vs core spec
====================================

``incompressibleFluid`` picks up models in two different ways. They
co-exist and serve distinct purposes; this page explains the design
rationale and when to use which.

What makes this hard
--------------------

OpenFOAM has effectively one mechanism for "what model is active":
read a string from a dictionary, look it up in a runtime-selection
table, instantiate. Selection, instantiation, and configuration are
all collapsed into the same ``addToRunTimeSelectionTable`` call.

That is fine for a model whose decision rule is "the user wrote a name
in a dictionary." It does not fit cleanly when the decision rule is
*either* of:

- "this model is optional, and its presence is signalled by the case
  layout" (e.g. a ``temperatureField`` file in ``0/`` enables
  Boussinesq).
- "exactly one of the following must run, and the choice depends on a
  case file the solver has already read" (e.g. PIMPLE vs SIMPLE vs
  PISO comes from ``system/fvSolution``).

NeoFOAM separates *is this model active?* (case-driven) from *how do
I instantiate it?* (Python type), and each axis warrants its own
mechanism.

The two discovery paths
-----------------------

.. mermaid::

   flowchart LR
       CASE[("Case directory<br/>system/<br/>constant/<br/>0/")]
       REGISTRY[["Plugin registry<br/>(@PluginSystem.register)"]]
       CORE[["Core specs<br/>(register_core_models)"]]

       CASE -->|"per-model<br/>@spec.detect"| PLUGIN["Plugin path<br/>optional models"]
       CASE -->|"detect_and_create()<br/>reads file once"| CORE_PATH["Core-spec path<br/>required component"]

       REGISTRY --> PLUGIN
       CORE --> CORE_PATH

       PLUGIN -->|"appended to"| OPTIONAL["LoadResult.optional_models"]
       CORE_PATH -->|"added to"| COREMODELS["LoadResult.core_models"]

       style CASE fill:#E3F2FD
       style REGISTRY fill:#FFF3E0
       style CORE fill:#FFF3E0
       style PLUGIN fill:#E8F5E9
       style CORE_PATH fill:#E8F5E9
       style OPTIONAL fill:#F3E5F5
       style COREMODELS fill:#F3E5F5

**1. Plugin registry — for optional, externally-pluggable physics.**

Models register themselves with a base class via
``@PluginSystem.register`` plus
``Model("...").register_with(<interface>)``. The solver discovers them
at startup with ``<interface>.detect_models(case_dir=...)``; each
model's own ``@spec.detect`` decides whether it should activate.

This is the right mechanism when:

- The solver works fine without the model.
- The decision to enable the model belongs to the case (file presence,
  dictionary key, etc.) rather than to a hard-coded list.
- Third-party packages might want to ship their own implementations.

In ``incompressibleFluid``, optional physics like ``boussinesq`` and
``spalart_allmaras`` use this mechanism. The call site is
``incompressibleFluidModel.detect_models()`` in ``create_fields.py``.

**2. Core specs — for required components with case-driven variant
selection.**

The solver registers a known list of specs via
``StagedInit.register_core_models([...])``. At LOAD time, exactly one
of them is selected — typically by reading a case file directly — and
added to ``LoadResult.core_models``. The solver pulls the chosen spec
out of ``state.core_models`` by index when building its execution
graph.

This is the right mechanism when:

- The component is mandatory; *some* variant has to run.
- The variants are known at solver authoring time (PIMPLE, SIMPLE,
  PISO).
- Selection depends on a case file that must be read by *something*
  authoritative, not by N independent ``@detect`` predicates racing
  each other.

In ``incompressibleFluid``, the pressure-velocity algorithm uses this
mechanism. ``PressureVelocityAlgorithm.detect_and_create()`` reads
``system/fvSolution`` exactly once and returns the right spec; the
``execution_graph`` step then pulls it out of
``self.state.core_models[0]``.

Decision tree
~~~~~~~~~~~~~

.. mermaid::

   flowchart TD
       START["I want to register a model"]
       Q1{"Is it mandatory<br/>(some variant<br/>must run)?"}
       Q2{"Do third parties<br/>need to ship<br/>their own?"}
       CORE["Core spec<br/>register_core_models([...])"]
       PLUGIN["Plugin registry<br/>@PluginSystem.register +<br/>register_with(...)"]

       START --> Q1
       Q1 -->|"yes"| CORE
       Q1 -->|"no"| Q2
       Q2 -->|"yes"| PLUGIN
       Q2 -->|"no"| PLUGIN

       style CORE fill:#E8F5E9
       style PLUGIN fill:#E8F5E9

If the answer to the first question is *yes*, you need a core spec —
plugins cannot enforce "exactly one of N must activate." Otherwise, a
plugin is the right tool whether or not third parties are involved.

A third path exists
-------------------

The framework also supports a YAML manifest (``models.yaml`` lists
``type`` + ``name`` + inline config per entry; loaded via
``load_manifest(manifest_path, case_dir, registry_name)``). It is the
right tool when the same model type appears multiple times with
different parameters (e.g. several heat sources).
``incompressibleFluid`` does not use it; the manifest path is
exercised in ``test/framework/integration/`` and is the natural
mechanism for future multi-source / multi-zone solvers.

Trade-offs
----------

The two paths trade off differently along two axes:

==============  =====================  =====================
Axis            Plugin                 Core specs
==============  =====================  =====================
Mandatory?      No (opt-in)            Yes (one of N)
Selection       Per-model ``@detect``  Solver decides
==============  =====================  =====================

Pinning everything to one mechanism would compromise on at least one
axis:

- *Plugin only:* mandatory components have no clean way to express
  "exactly one of these must activate"; each ``@detect`` has to know
  about the others.
- *Core specs only:* third parties cannot ship optional physics
  without modifying the solver.

The cost of two mechanisms is exactly what it sounds like — it is a
choice for the solver author, and that choice is part of the solver's
design. The mitigation is the decision tree above: *required and
case-decided* → core specs; *optional and case-detected* → plugin.

When this matters in practice
-----------------------------

The two mechanisms map to two concrete pieces of in-tree code:

- ``boussinesq`` — plugin path. Activated by a ``temperatureField``
  in ``0/``; the solver does not know it exists until ``detect_models``
  finds it.
- ``PressureVelocityAlgorithm`` — core-spec path. The solver registers
  ``PIMPLE``, ``SIMPLE``, and ``PISO`` specs at construction; LOAD
  picks one by reading ``system/fvSolution``.
