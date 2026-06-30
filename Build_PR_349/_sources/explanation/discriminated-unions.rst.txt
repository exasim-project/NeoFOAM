The discriminated-union plugin registry
=======================================

NeoFOAM's plugin system is built on Pydantic discriminated unions
wrapped in a runtime-extensible registry. This page explains why —
why discriminated unions, why a registry, and why both together.

What makes this hard
--------------------

OpenFOAM solves "which model implementation should I instantiate?"
with ``addToRunTimeSelectionTable``. The dictionary supplies a string;
the table maps it to a constructor. Subclasses register themselves at
static-init time. It is a well-understood pattern and it works.

What it does *not* offer:

- **Validation.** The string is checked for existence in the table;
  the configuration the string activates is not. Each solver has to
  re-implement its own argument parsing.
- **Schema export.** There is no way to ask "what are the valid
  choices for ``RAS.RASModel`` in this build?" without opening the
  source.
- **Type safety on the consumer side.** A ``tmp<RASModel>`` returned
  from ``RASModel::New`` does not tell the C++ compiler which subclass
  it actually is; downcasts are explicit and brittle.

A Python-first surface where input validation, UI generation, and
AI-assisted templating are first-class capabilities cannot rely on
"runtime string lookup with no schema." It needs a primitive that does
selection, validation, and schema export from the same definition.

Mechanism: discriminated unions plus a registry
-----------------------------------------------

Pydantic's discriminated union is a tagged-union type. A field of type
``Union[Cat, Dog, Lizard]`` with ``discriminator='pet_type'`` resolves
to the right subclass based on the value of ``pet_type`` in the input
dictionary:

.. code-block:: python

    class Cat(BaseModel):
        pet_type: Literal['cat']
        meows: int

    class Dog(BaseModel):
        pet_type: Literal['dog']
        barks: float

    class Model(BaseModel):
        pet: Union[Cat, Dog] = Field(discriminator='pet_type')

This gives type-safe selection: ``Model(pet={'pet_type': 'dog',
'barks': 3.14})`` constructs a ``Dog`` automatically and rejects
unknown ``pet_type`` values at validation time.

For NeoFOAM the primitive solves three problems at once:

- **Type-safe configuration loading.** A turbulence-model field in
  ``turbulenceProperties`` selects between ``kEpsilon``,
  ``SpalartAllmaras``, ``laminar``, etc. Discriminated unions encode
  that selection in the type system.
- **Schema generation.** ``BaseModel.model_json_schema()`` on a
  discriminated-union model produces a JSON schema with the
  discriminator mapping baked in. The same schema drives input
  validation, UI form generation, and AI-assisted case templating.
- **One source of truth.** The list of valid types is whatever is
  currently in the union — there is no separate "registry of names"
  that can drift from the type system.

The limitation: closed at definition time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pydantic's discriminated union must be defined at model creation time:

.. code-block:: python

    pet: Union[Cat, Dog, Lizard] = Field(discriminator='pet_type')

The members are baked in. You cannot add a fourth pet type from a
third-party package without rebuilding the parent model. That breaks
the goal of "third parties can ship their own physics models without
modifying core NeoFOAM code."

The registry: rebuild on registration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NeoFOAM's ``PluginSystem`` wraps the discriminated-union pattern in a
runtime-extensible registry. Each base class decorated with
``@PluginSystem.register(...)`` gets a registry of subclasses;
``@<Base>.register`` on a subclass appends to the registry **and**
rebuilds the parent's discriminated union to include the new member:

.. code-block:: python

    @PluginSystem.register(discriminator_variable="shape", discriminator="shape_type")
    class ShapeInterface(BaseModel):
        color: str

    @ShapeInterface.register
    class CircleConfig(BaseModel):
        shape_type: Literal["circle"]
        radius: float

    @ShapeInterface.register
    class SquareConfig(BaseModel):
        shape_type: Literal["square"]
        side: float

After both ``register`` calls, ``ShapeInterface.plugin_model`` is a
Pydantic model whose ``shape`` field is a discriminated union over
``Circle``, ``Square``. Add a third subclass and the union grows;
remove one and it shrinks.

.. mermaid::

   sequenceDiagram
       autonumber
       participant Mod as Plugin module
       participant Sub as Subclass<br/>(e.g. TriangleConfig)
       participant Reg as ShapeInterface<br/>registry
       participant Pyd as Pydantic<br/>schema cache
       participant Con as Consumer code

       Mod->>Sub: define class
       Mod->>Reg: @ShapeInterface.register
       Reg->>Reg: append Triangle to plugin_registry
       Reg->>Pyd: rebuild plugin_model<br/>(Union[Circle, Square, Triangle])
       Note over Pyd: cache invalidated for<br/>ShapeInterface.plugin_model

       rect rgb(232, 245, 233)
           Note over Con,Pyd: Correct path
           Con->>Reg: ShapeInterface.create({"shape_type": "triangle", ...})
           Reg->>Pyd: validate against<br/>current plugin_model
           Pyd-->>Con: Triangle instance
       end

       rect rgb(255, 235, 238)
           Note over Con,Pyd: Wrong path (footgun)
           Con->>Sub: ShapeInterface(...) directly
           Note over Con: Pydantic uses ShapeInterface's<br/>own schema, not the rebuilt union
           Sub-->>Con: validation error or<br/>silently wrong subclass
       end

The construction goes through a classmethod
(``ShapeInterface.create(...)`` or a plugin-specific helper) because
Pydantic caches schemas — instances have to be built from the
*current* ``plugin_model``, not from ``ShapeInterface`` itself.

Trade-offs
----------

The combined design has costs:

- **Import-order sensitivity.** ``register`` has a side effect on the
  parent model. Importing a third-party plugin package after a
  consumer has already constructed from the registry means the
  consumer was constructed against a stale union. In practice, this
  is contained to plugin-registration modules that always import
  their plugins before anything tries to construct from the registry,
  but it is a real footgun.
- **Indirect construction.** Users have to call
  ``<Base>.create(...)`` rather than ``<Base>(...)`` because the
  parent's schema cache would otherwise be stale. The error message
  when this is forgotten is not always obvious.
- **Schema cache invalidation.** Pydantic caches generated schemas
  per type. The framework rebuilds those caches when registration
  changes the union; subtle bugs are possible if a consumer captures
  ``plugin_model`` early and reuses it.

The OpenFOAM analogue (``addToRunTimeSelectionTable``) does not have
these footguns — it has different ones (no validation, no schema
export, no type safety). The trade is honest: register-rebuild gives
the schema-driven workflow at the cost of some import-order
discipline.

When this matters in practice
-----------------------------

The combination of discriminated union + registry is what lets NeoFOAM:

- **Validate input files against the currently-installed plugin set.**
  A typo in a turbulence-model name produces a Pydantic error that
  names the field, the bad value, and the valid alternatives — instead
  of a generic "model not found" from a runtime-selection table.
- **Generate a JSON schema that reflects every plugin available in the
  current environment**, third-party plugins included. This is what
  drives the schema-introspection capability documented in
  :doc:`schema-introspection`.
- **Avoid carrying a "string name → factory" dictionary in parallel
  with the type system.** The discriminator value *is* the type
  identifier. Renames stay consistent because there is only one place
  to rename.

In ``incompressibleFluid``, the plugin interface is
``incompressibleFluidModel`` (see
``src/neofoam/solver/incompressibleFluid/models/incompressibleFluidModel.py``).
Boussinesq and Spalart–Allmaras register against it; the registry's
discriminated union is what
``StagedInit.solver_inputs()`` walks during schema introspection.

For the two ways ``incompressibleFluid`` actually picks up registered
models at startup (plugin registry vs core specs), see
:doc:`discovery-mechanisms`.
