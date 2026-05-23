Solver and Models
=================

.. note::

   This page is retained as a redirect. The framework's solver/model
   authoring guide (decorators, 3-stage initialization, execution graph,
   model discovery, verification) lives in :doc:`initialization`.

Where to look
-------------

- **Spec / Runtime architecture** (``ModelSpec``, ``SolverSpec``, ``Model()``,
  ``Solver()``): :doc:`initialization` → *Architecture Overview*.
- **3-stage initialization** (LOAD / RESOLVE / VERIFY / BUILD):
  :doc:`initialization` → *3-Stage Initialization*.
- **Operations and the execution graph** (decorators, ``Operation``,
  ``Operations``, ``StepBuilder``, ``DAGResolver``):
  :doc:`components` and :doc:`initialization` → *Operations & Execution Graph*.
- **Model discovery** (plugin / core specs / YAML manifest, ``@detect``):
  :doc:`initialization` → *Model Discovery & Manifests*.
- **Configuration files and IO strategies**: :doc:`config_file`.
- **Plugin system internals** (discriminated unions, registry):
  :doc:`pluginsystem`.
