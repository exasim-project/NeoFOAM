neofoam.framework.solver
========================

.. automodule:: neofoam.framework.solver

The ``solver`` package implements the Spec/Runtime split for solvers:
``SolverSpec`` is the immutable, decorator-driven definition registered
at module import; ``SolverRuntime`` is the per-instance object created
by ``SolverSpec.instantiate(argv)``.

.. toctree::
   :maxdepth: 1

   spec
   runtime
