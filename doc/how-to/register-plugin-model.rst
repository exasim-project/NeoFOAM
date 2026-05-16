Register a plugin model with ``incompressibleFluid``
====================================================

Goal: extend ``incompressibleFluid`` with a new physics model that is
auto-detected from the case directory and slotted into the operation graph
without modifying solver code.

Prerequisites
-------------

- A ``BaseConfig`` for your model's parameters (see
  :doc:`/auto_tutorials/example_04_configure_with_io`).
- A clear answer to "when should this model activate?" — usually a file
  presence check or a dictionary key.

Steps
-----

1. **Create the spec and bind it to the plugin interface.**

   ``Model("Name").register_with(incompressibleFluidModel)`` makes the model
   discoverable by ``incompressibleFluidModel.detect_models()``.

   .. code-block:: python

       from pathlib import Path
       from typing import Annotated, Any
       from neofoam.framework.model import Model
       from neofoam.framework.context import FieldUpdates
       from neofoam.framework.initialization import ConfigContext, field
       from neofoam.solver.incompressibleFluid.models.incompressibleFluidModel import (
           incompressibleFluidModel,
       )

       my_model = Model("my_model").register_with(incompressibleFluidModel)

2. **Tell the framework when to activate the model.** ``@spec.detect``
   receives the case directory and returns ``bool`` (or ``list[str]`` of
   instance IDs for multi-instance models):

   .. code-block:: python

       @my_model.detect
       def detect(case_dir: Path) -> bool:
           return (case_dir / "constant" / "myModelProperties").exists()

3. **Load configuration.** Either register a config class with
   ``@spec.config`` and let the framework auto-construct from the manifest
   entry, or write a custom ``@spec.load`` that parses files directly:

   .. code-block:: python

       from neofoam.io import BaseConfig

       class MyModelConfig(BaseConfig):
           strength: float = 1.0

       @my_model.load
       def load(case_dir: Path, _entry: Any) -> MyModelConfig:
           return MyModelConfig.load(case_dir=case_dir)

4. **(Optional) Resolve cross-model dependencies.** ``@spec.resolve`` runs
   after every model is loaded. Use it to set flags on other models:

   .. code-block:: python

       @my_model.resolve
       def resolve(self: Any, ctx: ConfigContext) -> None:
           pressure = ctx.get("Pimple") or ctx.get("Simple") or ctx.get("Piso")
           if pressure is not None:
               pressure.my_model_active = True

5. **Build runtime fields.** ``@spec.build`` returns a list of ``InitStep``
   objects that the framework topologically sorts and executes:

   .. code-block:: python

       import pybFoam as pyf

       @my_model.build
       def build(self: Any, cfg: MyModelConfig) -> list[Any]:
           def create_phi_my(context: dict[str, Any]) -> Any:
               mesh = context["mesh"]
               return pyf.volScalarField.read_field(mesh, "phi_my")
           return [field("phi_my", create_phi_my, depends_on=["mesh"])]

6. **Add operations.** Operations declare ``operation_number`` (used for
   dependency-free ordering) and optionally ``depends_on`` to anchor into
   the existing solver graph:

   .. code-block:: python

       @my_model.operation(operation_number="2.6", depends_on=["momentum"])
       def apply_force(
           self: Any,
           U: Annotated[Any, "fields"],
           cfg: MyModelConfig,
       ) -> FieldUpdates:
           # ... mutate U
           return FieldUpdates({"U": U})

   To declare which OpenFOAM scheme/solver entries this operation needs at
   runtime, see :doc:`declare-fvschemes-requirements`.

Verify
------

With the model importable from your project, run any case that satisfies
``detect``. ``incompressibleFluid`` will pick it up automatically — no
changes to the solver:

.. code-block:: bash

    uv run python -m neofoam.solver.incompressibleFluid /path/to/case

To double-check the model was discovered without running the solver, see
:doc:`validate-without-running`.

Multi-instance variant
----------------------

If your model can be instantiated multiple times with different parameters,
return ``list[str]`` from ``@detect`` (the IDs become per-instance suffixes
on operation names) and use ``self.name`` inside ``@build`` /
``@operation`` to distinguish instances.
