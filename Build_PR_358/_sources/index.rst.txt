.. NeoFOAM documentation master file, created by
   sphinx-quickstart on Sat Dec 16 15:22:11 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to NeoFOAM!
===================

NeoFOAM is the coupling interface between OpenFOAM and NeoN. It provides
platform-portable implementations of common CFD algorithms and solvers using
NeoN as a computational backend, while leveraging standard OpenFOAM cases so
existing simulations can run on accelerator devices.

The documentation is organised by purpose. Pick the quadrant that matches
what you're trying to do:

- **Tutorials** — step-by-step lessons for newcomers. Start here if you're
  new to NeoFOAM.
- **How-to guides** — focused recipes for one specific task each. Use these
  when you already know what you're trying to do.
- **Reference** — module-by-module API description for the framework
  packages, generated from docstrings.
- **Explanation** — the *why* behind the design: architecture, three-stage
  initialization, plugin system, operation graph.

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   auto_tutorials/example_01_run_incompressible_fluid
   auto_tutorials/example_02_passive_scalar_plugin
   auto_tutorials/example_03_build_a_solver
   auto_tutorials/example_04_configure_with_io
   tutorials/scaffold-a-case-wizard

.. toctree::
   :maxdepth: 1
   :caption: How-to guides

   how-to/install
   how-to/declare-fields
   auto_how-to/example_register_a_model
   auto_how-to/example_add_an_init_step_category
   auto_how-to/example_work_with_config_files
   auto_how-to/example_custom_solver_with_configs
   auto_how-to/example_per_model_fvschemes
   auto_how-to/example_collect_and_save_configs
   auto_how-to/example_visualize_the_operation_dag
   auto_how-to/example_use_depends_for_injection
   auto_how-to/example_use_an_interface
   auto_how-to/example_trace_solver_performance

.. toctree::
   :maxdepth: 2
   :caption: Reference

   reference/framework/index
   reference/graph/index
   reference/initialization/index
   reference/model/index
   reference/solver/index

.. toctree::
   :maxdepth: 1
   :caption: Explanation

   explanation/goals-and-core-concepts
   explanation/solver-structure
   explanation/model-structure
   explanation/spec-runtime
   explanation/three-stage-init
   explanation/operations-and-the-dag
   explanation/parameter-injection
   explanation/discriminated-unions
   explanation/discovery-mechanisms
   explanation/schema-introspection

.. toctree::
   :maxdepth: 1
   :caption: Project

   ci
   debugProfileTools

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
