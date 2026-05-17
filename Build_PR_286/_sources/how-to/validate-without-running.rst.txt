Validate a case without running the solver
==========================================

Goal: check that a case's ``system/fvSchemes``, ``system/fvSolution``, and
model configs satisfy every active operation's requirements, before
spending time on a real solve.

Use ``StagedInit.validate``
---------------------------

``StagedInit.validate()`` runs LOAD → RESOLVE → VERIFY and returns a list
of ``VerificationError``. It does **not** run BUILD (no fields are
allocated, no mesh is loaded), and it does not raise on failure.

.. code-block:: python

    from pathlib import Path
    from neofoam.solver.incompressibleFluid.create_fields import create_init

    init = create_init(case_dir=Path("./pitzDaily"))
    errors = init.validate()

    if errors:
        for e in errors:
            print(f"{e.file_name}: {e.field} — {e.message}")
    else:
        print("Case valid for incompressibleFluid")

A clean run reports zero errors; otherwise each ``VerificationError``
carries ``file_name``, ``field``, ``error_type``, ``message``, and (when
relevant) ``subdict`` and ``input_value``.

Inspect what the solver expects
-------------------------------

Two introspection helpers expose the typed surface of the solver:

.. code-block:: python

    # Every config class the solver might consume
    schema_by_model = {
        name: cls.model_json_schema()
        for name, cls in init.solver_inputs().items()
    }

    # The exact fvSchemes the solver needs, as a typed Pydantic model
    schemes_schema = init.scheme_inputs().model_json_schema()

These are the entry points for UI generation, AI-assisted case templating,
and machine-checked input validation.

When to use this
----------------

- In CI, before pushing a case.
- In an editor or notebook, while authoring a new model.
- As the first step of a debugging session — most "solver crashed at
  startup" failures are missing scheme entries that ``validate()`` catches
  first.
