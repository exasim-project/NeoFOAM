Model structure
===============

A NeoFOAM plugin **model** is a typed configuration plus a set of
lifecycle hooks plus a set of operations. The solver declares the
loop topology and its own structural operations; a model declares
how to activate itself, what fields it needs, and which operations
it contributes. The framework wires the model into the solver at
LOAD time and merges its operations into the standard graph at
runtime.

This page mirrors :doc:`solver-structure` for the model side of the
contract. The plugin-registration mechanism is covered on
:doc:`discriminated-unions`; the activation and selection rules on
:doc:`discovery-mechanisms`; the operation merger on
:doc:`operations-and-the-dag`.

The model declaration and its config
------------------------------------

A model is a ``Model("name")`` instance attached to a solver's
plugin interface. Its typed configuration is a Pydantic
``BaseConfig`` subclass. From
``src/neofoam/solver/incompressibleFluid/models/boussinesq.py``:

.. code-block:: python

    class BoussinesqConfig(BaseConfig):
        beta: float = 3e-3
        TRef: float = 300.0
        Pr: float = Field(default=0.7, gt=0)
        Prt: float = Field(default=0.85, gt=0)
        hRef: float = 0.0

    boussinesq = Model("boussinesq").register_with(incompressibleFluidModel)

- ``BoussinesqConfig`` lists the model's typed inputs. The Pydantic
  validators (``gt=0``) catch bad values at LOAD time.
- ``Model("boussinesq")`` creates the spec; ``.register_with(...)``
  attaches it to ``incompressibleFluidModel``, the plugin interface
  exposed by the ``incompressibleFluid`` solver. Other solvers
  expose other plugin interfaces.

The plugin interface is what makes the registry rebuild work
(:doc:`discriminated-unions`); the solver finds registered models
with ``incompressibleFluidModel.detect_models()`` at LOAD
(:doc:`discovery-mechanisms`).

How a model integrates with the lifecycle
-----------------------------------------

A model declares four lifecycle hooks via decorators on the
``Model`` instance. They map directly onto the three-stage init
pipeline (see :doc:`three-stage-init`).

.. code-block:: python

    @boussinesq.detect
    def detect_model(_case_dir: Path) -> bool:
        try:
            props = pyf.dictionary.read("constant/transportProperties")
            return props.found("beta") and props.found("TRef")
        except Exception:
            return False

    @boussinesq.load
    def load(_case_dir: Path, _entry: Any) -> BoussinesqConfig:
        return _read_boussinesq_config()

    @boussinesq.resolve
    def resolve(self: Any, ctx: ConfigContext) -> None:
        for algo_name in ("Pimple", "Simple", "Piso"):
            pressure_model = ctx.get(algo_name)
            if pressure_model is not None:
                pressure_model.use_boussinesq = True
                return

    @boussinesq.build
    def build(self: Any, configs: BoussinesqConfig) -> list[object]:
        return [
            field("T", create_T, depends_on=["mesh"]),
            field("alphat", create_alphat, depends_on=["mesh"]),
            field("rhok", create_rhok, depends_on=["mesh", "fields.T"]),
            field("gh", create_gh, depends_on=["mesh"]),
            field("ghf", create_ghf, depends_on=["mesh"]),
            field("p_rgh", create_p_rgh,
                  depends_on=["mesh", "fields.p", "fields.rhok", "fields.gh"]),
        ]

What each hook does:

- ``@model.detect`` — runs first, decides whether the model should
  activate. Inspects the case directory (here:
  ``constant/transportProperties``). If it returns ``False``, the
  model is skipped — the solver runs as if the plugin weren't
  installed.
- ``@model.load`` — reads the case files into a typed config (here
  ``BoussinesqConfig``). No interaction with other models.
- ``@model.resolve`` — wires this model to peers through
  ``ConfigContext``. Boussinesq uses it to flip
  ``use_boussinesq=True`` on whichever pressure-velocity algorithm
  is active — *before* BUILD allocates fields, so the algorithm
  picks the Boussinesq variant of ``momentum`` / ``continuity``.
  ``@resolve`` is optional; a model with no cross-model wiring
  omits it.
- ``@model.build`` — returns a list of ``InitStep`` objects
  (typically ``field(...)`` declarations) the framework should
  execute. The ``depends_on`` arguments form a dependency graph the
  framework topologically sorts before executing — so ``rhok`` is
  created after ``T``, and ``p_rgh`` after ``p``, ``rhok``, and
  ``gh``.

Detect → load → resolve → build run in that order. The rationale
for splitting them — especially RESOLVE between LOAD and BUILD — is
on :doc:`three-stage-init`.

How a model contributes operations
----------------------------------

After BUILD has allocated the model's fields, the model is ready
to contribute operations to the solver loop. Operations are
declared the same way the solver declares its own, just with the
model's decorator:

.. code-block:: python

    @boussinesq.operation(operation_number="2.5", depends_on=["momentum"])
    @fvSchemes.add(ddt="default", div="div(phi,T)", laplacian="default")
    @fvSolution.add("T")
    def solve_energy(
        self: Any,
        T: volScalarField,
        phi: surfaceScalarField,
        turbulence: Annotated[ThermalTurbulenceModel, "models"],
        alphat: volScalarField,
    ) -> FieldUpdates:
        ...
        return FieldUpdates({"T": T, "alphat": alphat})

    @boussinesq.operation(operation_number="2.7", depends_on=["solve_energy"])
    def update_rhok(self: Any, T: volScalarField, rhok: volScalarField) -> FieldUpdates:
        ...
        return FieldUpdates({"rhok": rhok})

The decorator trio is the same one described in
:doc:`solver-structure` (``@operation``, ``@fvSchemes.add``,
``@fvSolution.add``); the parameter-injection rules are on
:doc:`parameter-injection`; ``depends_on`` and
``operation_number`` tell the merger where each operation should
land relative to others.

The operations registered on a model are what ``execution_graph``
picks up via ``self.state.optional_models`` and hands to the DAG
resolver. ``solve_energy`` lands between ``momentum`` (op_number
2.1) and ``continuity`` (op_number 2.2) because ``operation_number
= "2.5"`` falls between them; the precise placement rules are on
:doc:`operations-and-the-dag`.

What a complete model looks like
--------------------------------

End to end, a model is:

#. A ``BaseConfig`` subclass listing its typed inputs.
#. A ``Model("name").register_with(<plugin_interface>)``
   declaration.
#. ``@detect`` and ``@load`` lifecycle hooks (required).
#. ``@resolve`` and ``@build`` lifecycle hooks (optional —
   ``@resolve`` only when the model has cross-model wiring;
   ``@build`` only when the model allocates fields).
#. Zero or more ``@operation(...)`` decorators registering the
   operations the model contributes.

That is the contract a model author has to fill in. Everything
else — discovery, validation, merger, execution — is the
framework's job.

For a worked tutorial, see
:doc:`/auto_tutorials/example_02_passive_scalar_plugin`.
