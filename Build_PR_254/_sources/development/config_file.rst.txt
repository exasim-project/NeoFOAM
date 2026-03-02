Configuration File IO
=====================

NeoFOAM simulations rely on structured data/dictionaries to define solver settings, boundary conditions, and physical models.
Manually parsing, validating, and writing these files is repetitive and error-prone.

The ``neofoam.io`` module builds on `Pydantic <https://docs.pydantic.dev/>`_ to solve this.
Pydantic was chosen because it provides:

* **Declarative field definitions** — types and constraints (e.g. ``gt=0``, ``le=100``) are specified inline, not in a separate validation logic.
* **Automatic type coercion and error reporting** — invalid data produces clear, structured error messages with field names and expected types.
* **Serialisation built in** — ``model_dump()`` / ``model_validate()`` handle conversion between Python objects and dictionaries with no boilerplate.
* **Ecosystem compatibility** — Pydantic models integrate naturally with FastAPI, CLI tools, and other libraries.

On top of Pydantic, ``neofoam.io`` adds file-backed IO with a simple three-step workflow:

1. **Define** a Pydantic model with the fields you need.
2. **Decorate** it with ``@IOStrategy(...)`` to bind a file format and an optional subdict path.
3. **Use** ``load()`` / ``save()`` — the framework handles reading, writing, subdict isolation, and validation.


Quick Start
-----------

.. code-block:: python

   from pydantic import Field
   from neofoam.io import BaseConfig, YAML, IOStrategy

   @IOStrategy(YAML("solver_config.yaml"))
   class SolverConfig(BaseConfig):
       tolerance: float = Field(gt=0)
       maxIterations: int = Field(gt=0, le=10000)
       solver: str

   # Load, modify, save
   cfg = SolverConfig.load(case_dir="path/to/case")
   cfg.maxIterations = 500
   cfg.save(case_dir="path/to/case")


Loading
-------

``case_dir`` defaults to ``"."`` (current working directory), so ``load()`` with no arguments reads from the current directory.

.. code-block:: python

   # Load from current directory
   cfg = SolverConfig.load()

   # From a case directory (uses the registered filename)
   cfg = SolverConfig.load(case_dir="path/to/case")

   # Override filename at load time
   cfg = SolverConfig.load(case_dir="path/to/case", file="alternate.yaml")

   # Skip validation (useful for inspection)
   cfg = SolverConfig.load(case_dir="path/to/case", validate=False)


Saving
------

``case_dir`` defaults to ``"."`` as well.

.. code-block:: python

   # Save to current directory
   cfg.save()

   # Save to a specific case directory
   cfg.save(case_dir="path/to/case")

   # Override filename at save time
   cfg.save(case_dir="path/to/case", file="alternate.yaml")

When using **subdicts**, ``save()`` performs a **partial update** — it merges only the relevant section and preserves all other sections in the file.


Validation
----------

**On load (default)** — ``load()`` validates and raises ``pydantic.ValidationError`` on failure.

**Deferred** — load without validation, then check later:

.. code-block:: python

   cfg = SolverConfig.load(case_dir="case", validate=False)
   errors = cfg.check_validation()  # returns list[ValidationErrors]

**Batch** — validate multiple configs at once:

.. code-block:: python

   from neofoam.io import validate_models

   errors = validate_models([cfg1, cfg2, cfg3])

**Non-raising** — ``collect_errors()`` catches all exceptions and returns them as ``ValidationErrors``:

.. code-block:: python

   errors = SolverConfig.collect_errors(case_dir="case")


IO Strategies
-------------

The ``@IOStrategy`` decorator accepts a strategy configuration.
Each strategy implements the ``ReadingStrategy`` and ``WritingStrategy`` interface.

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Helper
     - Format
     - Example
   * - ``YAML(file)``
     - YAML
     - ``YAML("config.yaml")``
   * - ``YAML(file, subdict=...)``
     - YAML
     - ``YAML("fvSolution.yaml", subdict="PIMPLE")``
   * - ``JSON(file)``
     - JSON
     - ``JSON("config.json")``
   * - ``JSON(file, subdict=...)``
     - JSON
     - ``JSON("config.json", subdict="services.db")``
   * - ``OPENFOAM(file)``
     - OpenFOAM dictionary
     - ``OPENFOAM("system/fvSolution")``
   * - ``OPENFOAM(file, subdict=...)``
     - OpenFOAM dictionary
     - ``OPENFOAM("system/fvSolution", subdict="PISO")``

YAML, JSON, and OpenFOAM strategies support three subdict modes:

1. **No subdict** — the entire file maps to the model.
2. **Flat subdict** (``subdict="PIMPLE"``) — a top-level key.
3. **Nested subdict** (``subdict="solvers.p"``) — dot-separated path for deeper nesting.

.. code-block:: yaml

   # fvSolution.yaml
   PIMPLE:
     nOuterCorrectors: 2
   solvers:
     p:
       solver: PCG
       tolerance: 1e-6

.. code-block:: python

   @IOStrategy(YAML("fvSolution.yaml", subdict="PIMPLE"))
   class PIMPLEConfig(BaseConfig):
       nOuterCorrectors: int

   @IOStrategy(YAML("fvSolution.yaml", subdict="solvers.p"))
   class PSolverConfig(BaseConfig):
       solver: str
       tolerance: float

     @IOStrategy(OPENFOAM("system/fvSolution", subdict="PISO"))
     class PISOConfig(BaseConfig):
       nCorrectors: int
       nNonOrthogonalCorrectors: int


Extending — Adding a New Format
--------------------------------

To add support for a new file format, implement the ``ReadingStrategy`` and ``WritingStrategy`` protocols:

.. code-block:: python

   @runtime_checkable
   class ReadingStrategy(Protocol):
       def read(self, path: Any, encoding: str = "utf-8") -> dict[str, Any]: ...

   @runtime_checkable
   class WritingStrategy(Protocol):
       def write(self, data: dict[str, Any], path: Any, encoding: str = "utf-8") -> None: ...

**Example — TOML strategy:**

.. code-block:: python

   from pathlib import Path
   from typing import Any

   class TOMLStrategy:
       """TOML reading and writing strategy."""

       def read(self, path: Path, encoding: str = "utf-8") -> dict[str, Any]:
           import tomllib
           if not path.exists():
               raise FileNotFoundError(f"Configuration file not found: {path}")
           with open(path, "rb") as f:
               return tomllib.load(f)

       def write(self, data: dict[str, Any], path: Path, encoding: str = "utf-8") -> None:
           import tomli_w
           path.parent.mkdir(parents=True, exist_ok=True)
           with open(path, "wb") as f:
               tomli_w.dump(data, f)

Create a helper function and use it with ``@IOStrategy``:

.. code-block:: python

   def TOML(file):
       strategy = TOMLStrategy()
       return {
           "input_file": file,
           "reading_strategy": strategy,
           "writing_strategy": strategy,
       }

   @IOStrategy(TOML("settings.toml"))
   class AppConfig(BaseConfig):
       debug: bool
       log_level: str
