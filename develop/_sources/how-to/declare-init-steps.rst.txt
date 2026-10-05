Declare BUILD-stage init steps
==============================

The BUILD stage does not create runtime objects directly. A model's or a
solver's ``@build`` function returns a list of :class:`InitStep
<neofoam.framework.initialization.InitStep>` *recipes*; the framework sorts
them by the dependencies they declare and executes them in that order,
handing each one the objects the earlier steps produced. Declaring the
dependency instead of relying on call order is what keeps initialization
order correct as models are added and removed.

For *why* BUILD is a separate stage at all, see
:doc:`/explanation/three-stage-init`. On-disk ``0/<name>`` fields are not
declared here — they come from ``Model.field(...)`` and the framework
synthesises their steps for you (:doc:`declare-fields`). This page covers
everything ``@build`` still has to carry itself: computed fields, control
objects, meshes, runtimes and other side-effect setup.

The shape of a build function
-----------------------------

``@<spec>.build`` receives the model runtime and returns a list of steps.
Each step pairs a name with a closure that will later receive the
accumulated ``context`` dict:

.. code-block:: python

    from typing import Any

    import pybFoam as pyf

    from neofoam.framework.initialization import model


    @mrf.build
    def build(self: Any) -> list[Any]:
        def create_mrf_zones(context: dict[str, Any]) -> pyf.IOMRFZoneList:
            zones = pyf.IOMRFZoneList(context["mesh"])
            self.zones = zones
            return zones

        return [model("mrf_zones", create_mrf_zones, depends_on=["mesh"])]

Nothing is built when ``build()`` runs — it only collects recipes.
``create_mrf_zones`` fires later, at the point in the sorted order where
``mesh`` is already in the context.

Pick a helper
-------------

Four helpers build an ``InitStep``. They differ in the prefix they add to
the step name and in where the produced object ends up on the
:class:`~neofoam.framework.context.Context`:

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Helper
     - Step name
     - Ends up at
   * - ``field("U", …)``
     - ``fields.U``
     - ``ctx.fields["U"]``
   * - ``model("turbulence", …)``
     - ``models.turbulence``
     - ``ctx.models["turbulence"]``
   * - ``operator("momentum", …)``
     - ``operators.momentum``
     - ``ctx.models["momentum"]``
   * - ``lazy("mesh", …)``
     - ``mesh``
     - ``ctx.mesh``

``lazy`` adds no prefix and is the one to use for anything that is not a
field, a model or an operator. Two of its names are special: ``mesh``
populates ``ctx.mesh`` and ``time`` populates ``ctx.time``. Any other
``lazy`` name falls back to ``ctx.models`` and logs a warning — if you have
a whole family of such objects, register a category instead of living with
the warning (:doc:`/auto_how-to/example_add_an_init_step_category`).

Note that ``operator`` shares the ``models`` slot: there is no
``ctx.operators``.

Declare dependencies
--------------------

``depends_on`` lists the *step names* — the prefixed ones — that must run
first:

.. code-block:: python

    from neofoam.framework.initialization import field, lazy, model

    [
        lazy("runtime", create_runtime),
        lazy("mesh", create_mesh, depends_on=["runtime"]),
        field("phi", create_phi, depends_on=["fields.U"]),
        model("turbulence", create_turbulence,
              depends_on=["fields.U", "fields.phi"]),
    ]

The framework topologically sorts the whole collected list, so the order
you return them in does not matter. Independent steps are ordered
lexicographically by name, so two runs of the same case initialize in the
same order.

The graph is validated *before* any step executes: a duplicate step name, a
cycle, or a ``depends_on`` naming a step that nobody contributes raises
:class:`InitializationGraphError
<neofoam.framework.initialization.InitializationGraphError>` up front
rather than halfway through construction. A failure inside an initializer
is wrapped in :class:`InitStepExecutionError
<neofoam.framework.initialization.InitStepExecutionError>`, which reports
the step name and its dependencies.

Read the context
----------------

Inside an initializer, the ``context`` dict is keyed by **full step name**,
not by the Context slot the value later lands in — ``context["fields.U"]``,
not ``context["U"]``:

.. code-block:: python

    def create_pressure_reference(context: dict[str, Any]) -> PressureReference:
        mesh = context["mesh"]
        p = context["fields.p"]
        p_rgh = context.get("fields.p_rgh")   # optional: may not be contributed
        ...

Use ``context.get(...)`` for anything a plugin may or may not have
contributed, and keep it out of ``depends_on`` unless it is genuinely
required.

Three details worth knowing
---------------------------

``write=True``
    Only on ``field(...)``. Flags the field for persistence: its name is
    collected into ``ctx.write_fields`` and the write backend picks exactly
    those up at output time.

A leading underscore
    A step named ``_foam_time`` stays in the executor's working dict, so
    later steps can depend on it, but is never routed onto the
    ``Context``. Use it for init-only resources that must stay alive
    without being part of the public simulation state.

``replaces=[…]``
    Supersede another step. When a step carrying ``replaces=["mesh"]`` is
    present, the builder drops the default ``mesh`` step — this is how an
    in-process mesh-preprocessing pipeline stands in for the usual
    disk read. Naming a step that is not present is an error, so a typo
    cannot silently disable the replacement.

Test one step at a time
-----------------------

An initializer is a plain function of a dict, so a single step can be
exercised without running the pipeline — hand it exactly the keys it
declares and nothing else:

.. code-block:: python

    def test_turbulence_initializer():
        steps = {s.name: s for s in my_model.run_build()}
        step = steps["models.turbulence"]

        assert step.depends_on == ["fields.U", "fields.phi"]

        turbulence = step.initializer(
            {"fields.U": fake_velocity, "fields.phi": fake_flux}
        )
        assert turbulence is not None

If the step reads a context key it never declared, this test is what
catches it: the key is simply absent from the dict you passed.

.. seealso::

   - :doc:`declare-fields` — the declarative route for ``0/<name>`` fields.
   - :doc:`/explanation/three-stage-init` — why LOAD / RESOLVE / BUILD.
   - :doc:`/auto_how-to/example_add_an_init_step_category` — route a new
     category onto the ``Context``.
   - :doc:`/auto_how-to/example_use_depends_for_injection` — how operations
     receive what BUILD produced.
