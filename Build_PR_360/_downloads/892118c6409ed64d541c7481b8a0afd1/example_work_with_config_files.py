"""
Work with configuration files
=============================

NeoFOAM configs are :class:`BaseConfig` (Pydantic) subclasses tagged
with an ``@IOStrategy`` decorator that binds a *file* and a *format*
(YAML / JSON / OpenFOAM dictionary) to the class. The class then
gains ``.load()`` / ``.save()`` / ``.collect_errors()`` methods that
read and write that file with full type validation.

This page walks through the four things you typically do with
configs — declare, load, validate, and share a file between multiple
configs via subdicts. The YAML fixtures shipped alongside this
example are staged into a temporary case directory and loaded with
the real on-disk reader; the file contents shown below are read
straight from those fixtures by Sphinx.
"""

# %%
# Set up a throwaway case directory and locate the fixtures
# ---------------------------------------------------------
# Every example below stages real YAML files from
# ``examples/how-to/*.yaml`` into the same temporary directory. The
# real solver hands you a ``case_dir`` argument; here we make our
# own.  ``HERE`` resolves to the directory containing this script so
# every fixture stays addressable, including when sphinx-gallery
# ``exec()``s the script with no ``__file__``.

import inspect
import shutil
import tempfile
from pathlib import Path

from neofoam.io import BaseConfig, IOStrategy, YAML


def _here() -> None:
    """Marker used to locate this script on disk via inspect.getfile()."""


HERE = Path(inspect.getfile(_here)).resolve().parent
case_dir = Path(tempfile.mkdtemp(prefix="neofoam_howto_"))
print("case_dir =", case_dir)


# %%
# Declare a config class
# ----------------------
# ``@IOStrategy(YAML("...yaml"))`` binds the class to a file *and* a
# format in one decorator. The class body is a plain Pydantic model:
# field names, types, and default values become the schema you'll
# validate against. The decorator attaches an ``io_config`` class var
# the framework reads at ``load()`` time.


@IOStrategy(YAML("solver_config.yaml"))
class SolverConfig(BaseConfig):
    name: str
    dt: float
    end_time: float
    write_interval: int = 10


# Show what the binding looks like.
print("file:", SolverConfig.io_config.file)
print("reader:", type(SolverConfig.io_config.reader).__name__)
print("default path under case_dir:", SolverConfig.get_default_path(case_dir))


# %%
# Read a config from disk
# -----------------------
# In a real run the file already exists in the case directory — the
# user wrote it, or it shipped with the tutorial. The class method
# :meth:`BaseConfig.load` takes a ``case_dir`` and resolves the
# filename via ``io_config.file``, so you never spell the path twice.
#
# The fixture used here lives at ``examples/how-to/solver_config.yaml``:
#
# .. literalinclude:: ../../examples/how-to/solver_config.yaml
#    :language: yaml

shutil.copy(HERE / "solver_config.yaml", case_dir / "solver_config.yaml")

cfg = SolverConfig.load(case_dir=case_dir)
print("loaded:", cfg)
print("cfg.dt =", cfg.dt, "(type:", type(cfg.dt).__name__ + ")")


# %%
# Write changes back to disk
# --------------------------
# Once loaded, the instance behaves like any Pydantic model —
# mutate fields in memory, then :meth:`BaseConfig.save` writes the
# YAML back to the same path the loader read from.

cfg.write_interval = 50
cfg.save(case_dir=case_dir)

print("file after save:")
print((case_dir / "solver_config.yaml").read_text())


# %%
# Loading bad data raises with everything wrong at once
# -----------------------------------------------------
# Pydantic batches errors per call — one ``ValidationError`` reports
# every field that's wrong, not just the first. Useful when a user
# mistypes three keys in one file.
#
# The intentionally-broken fixture lives at
# ``examples/how-to/solver_config_bad.yaml``:
#
# .. literalinclude:: ../../examples/how-to/solver_config_bad.yaml
#    :language: yaml

shutil.copy(HERE / "solver_config_bad.yaml", case_dir / "solver_config.yaml")

try:
    SolverConfig.load(case_dir=case_dir)
except Exception as exc:
    print(type(exc).__name__, "raised — listing every problem:")
    for err in exc.errors():
        print(f"  {'.'.join(map(str, err['loc'])):12s} {err['msg']}")


# %%
# Validate without raising via collect_errors
# -------------------------------------------
# Use :meth:`BaseConfig.collect_errors` when you want a flat list of
# problems with file/subdict context — typical inside a CLI that
# shows the user every error before exiting. Returns ``[]`` on
# success. The framework's bulk validator
# (:func:`neofoam.io.validate_models`) is built on top of this.

errors = SolverConfig.collect_errors(case_dir=case_dir)
print(f"{len(errors)} error(s):")
for e in errors:
    field = ".".join(map(str, e.field)) if e.field else "<file>"
    print(f"  {field:12s} {e.error_type}: {e.message[:60]}")


# %%
# Share one file between multiple configs via subdicts
# ----------------------------------------------------
# A single ``fvSolution.yaml`` typically holds several disjoint
# blocks (``PIMPLE``, ``solvers.p``, ``solvers.U``). Each one gets
# its own ``BaseConfig`` class pointed at the same file with a
# different ``subdict`` path; ``.load()`` then only sees its
# own block.
#
# .. literalinclude:: ../../examples/how-to/fvSolution.yaml
#    :language: yaml

shutil.copy(HERE / "fvSolution.yaml", case_dir / "fvSolution.yaml")


@IOStrategy(YAML("fvSolution.yaml", subdict="PIMPLE"))
class PIMPLEConfig(BaseConfig):
    n_outer_correctors: int
    n_inner_correctors: int


@IOStrategy(YAML("fvSolution.yaml", subdict="solvers.p"))
class PressureSolver(BaseConfig):
    solver: str
    tolerance: float


pimple = PIMPLEConfig.load(case_dir=case_dir)
psolver = PressureSolver.load(case_dir=case_dir)

print("PIMPLE   :", pimple)
print("solvers.p:", psolver)
print("subdict path on PIMPLEConfig:", pimple.subdict)


# %%
# Other formats
# -------------
# Swap the strategy in the decorator to read the same data shape
# from a different format — no other changes needed. The three
# strategies that ship today:
#
# - :func:`~neofoam.io.YAML` — human-readable, supports subdicts,
#   round-trips floats safely.
# - :func:`~neofoam.io.JSON` — strict, machine-friendly, also
#   supports subdicts.
# - :func:`~neofoam.io.OF` — OpenFOAM dictionary format, lets you
#   point at an existing case's ``system/fvSolution`` or
#   ``constant/transportProperties`` directly.
#
# A typical migration looks like changing ``YAML("foo.yaml")`` to
# ``OF("foo")`` and pointing the case at the OpenFOAM-format file.


# %%
# See also
# --------
#
# - :doc:`/reference/framework/dependency_resolver` — ``BaseConfig``
#   parameters in operations are pulled from ``runtime.config`` via
#   the same resolver shown in
#   :doc:`example_use_depends_for_injection`.
# - :doc:`example_register_a_model` — the LOAD stage's job is to
#   instantiate one of these configs per model and return it inside
#   a :class:`LoadResult`.
