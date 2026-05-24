Parameter injection
===================

Operation bodies in NeoFOAM rarely take ``ctx: Context`` as their
only argument. Instead, they declare which fields and models they
need by name in the function signature, and the framework fills
those parameters before calling the operation. This page explains
how that filling works and what problem it solves.

.. seealso::
   For the merger that decides *which* operations run and in what
   order, see :doc:`operations-and-the-dag`. For the Context data
   class itself, see :doc:`solver-structure`.

What problem it solves
----------------------

A NeoFOAM solver runs dozens of operations contributed by the solver
author, the active pressure-velocity algorithm, and any number of
optional plugin models. None of these authors share a call site —
the framework calls each operation with the same machinery.

Two unattractive alternatives that the framework rejects:

- **Pass the Context only.** Every operation receives ``ctx`` and
  fishes its inputs out of ``ctx.fields[...]``. Works, but the
  operation's dependencies are buried inside the body rather than
  visible on the signature — making static analysis, IDE
  autocomplete, and validation harder.
- **Operations register their inputs separately.** A side-channel
  declares "``momentum`` needs ``U``, ``phi``, ``p``, ``turbulence``,
  ``pimple_control``". Works, but the function body and the
  side-channel can drift out of sync.

Parameter injection puts the inputs *on the signature itself*. The
framework reads the signature with ``inspect.signature`` and decides
where each parameter comes from. Adding a new dependency means
adding a parameter — nothing else changes.

How injection works
-------------------

When the framework calls an operation it walks each parameter and
applies these rules in order. The logic lives in
``src/neofoam/framework/operation_wrapper.py``.

.. list-table::
   :header-rows: 1
   :widths: 32 30 38

   * - Pattern
     - Source
     - Example
   * - ``self`` / ``cls``
     - passed through (Python method binding)
     - ``def increment_time(self, ...)``
   * - ``ctx: Context``
     - the full Context object
     - ``def increment_time(self, ctx: Context)``
   * - ``Annotated[T, "models"]``
     - ``ctx.models.get(<param_name>)``
     - ``turbulence: Annotated[TurbulenceModel, "models"]``
   * - ``Annotated[T, "fields"]``
     - ``ctx.fields.get(<param_name>)``
     - ``U: Annotated[volVectorField, "fields"]``
   * - ``Annotated[T, "<ns>"]``
     - ``getattr(ctx, "<ns>", {}).get(<param_name>)``
     - generic namespace lookup, used by framework extensions
   * - ``Annotated[T, Depends(...)]``
     - resolved through the ``Depends`` system
     - mainly used by ``@solver.initializer``
   * - Bare typed parameter
     - ``ctx.fields[<param_name>]`` if the name is present
     - ``U: volVectorField`` (auto-filled from ``ctx.fields["U"]``)

The parameter *name* is the lookup key. The parameter *type* is the
static contract the operation expects — the framework does not
validate the actual type against the declared type at injection
time.

Missing entries
~~~~~~~~~~~~~~~

If a ``"models"`` lookup misses (the case has no Boussinesq plugin
loaded, for example), the parameter arrives as ``None``. Operations
that take an optional model write
``Annotated[Optional[T], "models"]`` and guard with ``if model:``:

.. code-block:: python

    @incompressibleFluid.operation(depends_on=["continuity"])
    def turbulence_correction(
        self: Any,
        laminarTransport: Annotated[Optional[CorrectableModel], "models"],
        turbulence: Annotated[Optional[CorrectableModel], "models"],
    ) -> FieldUpdates:
        if laminarTransport:
            laminarTransport.correct()
        if turbulence:
            turbulence.correct()
        return FieldUpdates({})

If a bare typed parameter is *not* found in ``ctx.fields``, the
framework leaves the kwarg unset; Python then raises ``TypeError``
at call time. That is the right behaviour: a missing required field
is a programming error, not a nullability concern the operation
should handle.

Concrete examples
-----------------

The ``momentum`` operation uses both injection modes side by side
(``src/neofoam/solver/incompressibleFluid/models/pressure_velocity/pimpleAlgorithm.py``):

.. code-block:: python

    @pimple.operation(operation_number="2.1")
    def momentum(
        U: volVectorField,                                  # ctx.fields["U"]
        phi: surfaceScalarField,                            # ctx.fields["phi"]
        p: volScalarField,                                  # ctx.fields["p"]
        turbulence: Annotated[TurbulenceModel, "models"],   # ctx.models["turbulence"]
        pimple_control: Annotated[PimpleControl, "models"], # ctx.models["pimple_control"]
    ) -> FieldUpdates:
        ...

``set_time_step`` mixes the *full Context* style with model
injection (``incompressibleFluid.py``):

.. code-block:: python

    @incompressibleFluid.operation()
    def set_time_step(
        self: Any,
        ctx: Context,                                          # full Context
        pressure_velocity: Annotated[Optional[Any], "models"], # ctx.models["pressure_velocity"]
        cfl_condition: Annotated[Optional[Any], "models"],     # ctx.models["cfl_condition"]
    ) -> None:
        ...

Both styles are valid; the operation author picks whichever makes
the dependencies most visible.

Trade-offs
----------

- **Names are load-bearing.** Renaming the parameter ``U`` to
  ``velocity`` would break the auto-fill from ``ctx.fields["U"]``.
  Field and model names are part of the public contract of the
  operation; treat them as carefully as any other public identifier.
- **The Python type checker doesn't see the contract.** ``U:
  volVectorField`` is a hint about what the framework *should*
  inject, not a guarantee the framework will inject anything. If
  ``ctx.fields["U"]`` is missing, the runtime raises a ``TypeError``
  and the static type checker had no chance to warn.
- **Bare auto-fill only looks in ``ctx.fields``.** A parameter
  named ``turbulence`` without an ``Annotated[..., "models"]`` tag
  is looked up in ``ctx.fields`` and will miss — ``turbulence`` is
  a model, not a field. Tag consistently: bare = field,
  ``"models"`` = model.
- **Missing models arrive as None, missing required fields raise.**
  An asymmetry that follows from how the resolver writes back
  kwargs (``.get(name)`` for ``"models"``/``"fields"``, but skipped
  entirely for bare params that miss ``ctx.fields``). It is the
  pattern that lets *optional* models be declared with
  ``Optional[T]`` and the body guard with ``if model:``.

When this matters in practice
-----------------------------

- Reading any operation in
  ``src/neofoam/solver/incompressibleFluid/``: the signature tells
  you exactly what the operation depends on. No global registry
  inspection is needed.
- Writing a plugin model: declare its dependencies on the signature
  and the framework wires them up at the call site. You never
  touch ``execution_graph`` to hook the new operation up.
- Debugging a missing dependency: the ``TypeError`` from Python at
  call time names the parameter; that parameter name is the lookup
  key into ``ctx.fields`` or ``ctx.models``.

For how this connects to the merger and the surrounding loop, see
:doc:`operations-and-the-dag`.
