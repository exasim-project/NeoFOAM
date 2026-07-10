Declare on-disk fields with typed BCs
=====================================

Every CFD case carries ``0/<name>`` field files — ``U``, ``p``, ``T``,
``nut``, … — that pin the field's dimensions, internal value, and per-patch
boundary condition. NeoFOAM declares those files at model registration time
via :meth:`Model.field <neofoam.framework.model.spec.ModelSpec.field>`, so
the disk schema lives next to the runtime factory and shows up in
``configurations(solver)`` for a case author or an LLM to fill.

The big picture
---------------

A model owns *both* its dictionary configs (``constant/...``, ``system/...``)
*and* the ``0/<name>`` files it reads — declared by:

* :meth:`Model.config(...) <neofoam.framework.model.spec.ModelSpec.config>` —
  ``BaseConfig`` subclasses (decorated with ``@IOStrategy(OF("…"))``);
* :meth:`Model.field(...) <neofoam.framework.model.spec.ModelSpec.field>` —
  ``(name, dimensions, value_type, allowed_bcs)`` declarations;
  :func:`neofoam.fields.schema.schema_for` synthesises a per-field
  ``BaseConfig`` bound to ``0/<name>``.

``configurations(solver)`` surfaces both. The ``.fields`` property filters
to just the synthesised field schemas — ``.dicts`` excludes them.

Declare a field on a model
--------------------------

The PIMPLE algorithm declares its velocity and pressure fields like this:

.. code-block:: python

    from neofoam.fields import (
        FixedValueBC, GenericBC, NoSlipBC, Scalar, Vector, ZeroGradientBC,
    )
    from neofoam.framework.model.spec import Model

    pimple = Model("Pimple")

    pimple.field(
        "U",
        dimensions=[0, 1, -1, 0, 0, 0, 0],
        value_type=Vector,
        allowed_bcs=[NoSlipBC, FixedValueBC, GenericBC],
        write=True,
    )

    pimple.field(
        "p",
        dimensions=[0, 2, -2, 0, 0, 0, 0],
        value_type=Scalar,
        allowed_bcs=[FixedValueBC, ZeroGradientBC, GenericBC],
        write=True,
    )

The call returns a :class:`~neofoam.fields.decl.FieldDecl` for callers
that want to introspect the declaration (e.g. to share it across
specs), but binding the return value is optional — the side-effect on
``pimple`` is what makes the field part of the schema.

What each argument does:

``name``
    The OpenFOAM field name and the on-disk path key (``0/<name>``).

``dimensions``
    OpenFOAM dimension exponents ``[M, L, T, Θ, N, I, J]``. Pinned by the
    declaration; the case author cannot widen it.

``value_type``
    :class:`~neofoam.fields.value_types.Scalar` →
    ``volScalarField`` + ``"uniform 0"``;
    :class:`~neofoam.fields.value_types.Vector` →
    ``volVectorField`` + ``"uniform (0 0 0)"``.

``allowed_bcs``
    The discriminated union of BC arms permitted on this field's
    ``boundaryField``. Include :class:`~neofoam.fields.bc.GenericBC` if
    you want unknown BCs (``codedFixedValue``, wall functions, …) to
    parse through the smart-union fallback instead of failing.

``write``
    Forwarded to the runtime ``InitStep``: persist the field at
    ``write_output`` time.

That's the whole declaration. The framework synthesises the matching
runtime :class:`InitStep` automatically — ``@pimple.build`` only carries
the model's *non-field* bits (control objects, computed/intermediate
fields with no ``0/<name>`` file, side-effect-only setup steps). If you
need a side effect to run *before* the auto-synthesised read fires
(e.g. ``mesh.setFluxRequired("p_rgh")``), put it in a small
``lazy("p_rgh_flux_required", …)`` step in ``@build`` and list that
step's name in the field's ``depends_on``. The topological sort handles
ordering.

Read and write a case
---------------------

From a case directory, load every declared field at once:

.. code-block:: python

    from neofoam.fields import load_fields, save_fields
    from neofoam.solver.incompressibleFluid.incompressibleFluid import incompressibleFluid

    fields = load_fields("path/to/case", solver=incompressibleFluid)
    fields["U"].boundaryField["inlet"]   # → a typed BC arm

    # Mutate freely:
    fields["U"].boundaryField["inlet"] = FixedValueBC(value=[1.0, 0.0, 0.0])
    save_fields(fields, "path/to/case")

Add a new typed BC arm
----------------------

The bundled arms (``NoSlipBC``, ``FixedValueBC``, ``ZeroGradientBC``) cover
the most common patches; anything else routes through
:class:`~neofoam.fields.bc.GenericBC` and survives the round-trip with its
extras intact. To upgrade an extra to a typed arm, add a ``BaseModel``
subclass with a ``type: Literal["…"]`` discriminator and list it in the
relevant field's ``allowed_bcs``:

.. code-block:: python

    class FixedFluxPressureBC(BaseModel):
        type: Literal["fixedFluxPressure"] = "fixedFluxPressure"
        rho: str
        value: str

    boussinesq.field(
        "p_rgh",
        dimensions=[0, 2, -2, 0, 0, 0, 0],
        value_type=Scalar,
        allowed_bcs=[FixedValueBC, FixedFluxPressureBC, GenericBC],
    )

The Boussinesq case
-------------------

Models can add fields to a union without modifying the model that
declared the original pair. The ``boussinesq`` plugin declares ``p_rgh``,
``T``, and ``alphat`` on top of pimple's ``U`` / ``p`` — the disk schema
ends up with ``{U, p, p_rgh, T, alphat}`` only when boussinesq is
detected. No source case needs to be mirrored; ``save_fields`` writes the
union, ``load_fields`` reads it back.
