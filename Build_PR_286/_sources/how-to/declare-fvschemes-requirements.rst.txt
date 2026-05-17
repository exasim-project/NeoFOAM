Declare fvSchemes / fvSolution requirements on an operation
============================================================

Goal: tell the framework which ``system/fvSchemes`` and ``system/fvSolution``
entries an operation needs at runtime, so that missing or invalid entries
fail fast at startup instead of crashing mid-solve.

The decorators
--------------

Two decorators in ``neofoam.foam`` annotate an operation with its
requirements:

.. code-block:: python

    from neofoam.foam import fvSchemes, fvSolution
    from neofoam.framework.context import FieldUpdates

    @pimple.operation(operation_number="2.1")
    @fvSchemes.add(
        ddt="ddt(U)",
        div="div(phi,U)",
        grad="grad(U)",
        laplacian="laplacian(nuEff,U)",
    )
    @fvSolution.add("U")
    def momentum(...) -> FieldUpdates:
        ...

Short names map to OpenFOAM section names:

==================  ================================
Short name          OpenFOAM section
==================  ================================
``ddt``             ``ddtSchemes``
``div``             ``divSchemes``
``grad``            ``gradSchemes``
``laplacian``       ``laplacianSchemes``
``snGrad``          ``snGradSchemes``
``interpolation``   ``interpolationSchemes``
==================  ================================

Unknown short names pass through unchanged:
``@fvSchemes.add(wallDist="method")`` adds a ``wallDist`` requirement
verbatim.

When verification runs
----------------------

The framework runs verification automatically between the RESOLVE and BUILD
stages of ``StagedInit.run``, provided the ``LoadResult`` returned from
``@init.load`` populates ``fv_schemes_config`` and/or ``fv_solution_config``:

.. code-block:: python

    from neofoam.foam.fv_configs import FvSchemesConfig, FvSolutionConfig
    from neofoam.framework.initialization import LoadResult, StagedInit

    @init.load
    def load_config() -> LoadResult:
        return LoadResult(
            core_models=[...],
            optional_models=[...],
            fv_schemes_config=FvSchemesConfig.load(case_dir=case_dir, validate=False),
            fv_solution_config=FvSolutionConfig.load(case_dir=case_dir, validate=False),
        )

If you don't pass these in, requirements are still recorded on each
operation but no automatic check is performed.

What verification does
----------------------

For every active operation, the framework:

1. Confirms the entry exists in the corresponding dictionary (structural
   check). For example, ``@fvSchemes.add(div="div(phi,U)")`` requires
   ``divSchemes/div(phi,U)`` to be present.
2. Validates the entry value against a typed Pydantic scheme model
   (``DdtScheme``, ``DivScheme``, etc.) where one applies.

On failure, ``StagedInit.run`` raises ``RuntimeError`` listing every error
with file, field, and message.

To check a case without actually running the solver, see
:doc:`validate-without-running`.

Multiple operations sharing requirements
----------------------------------------

Both decorators are stackable — register multiple times and the
requirements accumulate. The framework deduplicates by ``(section, key)``
and ``field`` respectively, so it's safe for two operations to declare the
same requirement.
