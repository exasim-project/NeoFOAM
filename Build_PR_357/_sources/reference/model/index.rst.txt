neofoam.framework.model
=======================

.. automodule:: neofoam.framework.model

The ``model`` package implements the Spec/Runtime split for models:
``ModelSpec`` is the immutable, decorator-driven definition registered
at module import; ``ModelRuntime`` is the per-instance object created
by ``ModelSpec.instantiate(case_dir, instance_id)``.

.. toctree::
   :maxdepth: 1

   spec
   runtime
