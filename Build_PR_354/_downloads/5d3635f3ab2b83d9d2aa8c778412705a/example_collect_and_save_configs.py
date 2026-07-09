"""
Collect and save all of a solver's configs
===========================================

After a solver's LOAD and RESOLVE init stages, it knows every config it
consumes. This page shows how to *collect* those configs from the
:doc:`incompressibleFluid </auto_tutorials/example_01_run_incompressible_fluid>`
solver, *print* them, and *save* them back to disk as a fresh case
directory.

Three handles drive this:

- ``neofoam.configurations(solver)`` — the **case-free** schema view: every
  config class the solver may consume, derived from what the solver declares
  on its spec at import. No case directory needed.
- ``LoadResult.configs`` — the loaded config **instances** (values read
  from a specific case).
- ``LoadResult.config_classes`` — the declared config **classes** for that
  load, including classes a model registers but never loads.

Saving the instances with ``neofoam.io.save_configs`` (or the view's
``.save(...)``) reconstructs the case files on disk. Filling fresh instances
from the schema and saving them is how a case can be created from configs
alone — e.g. by an agent that knows the schema but starts from an empty
directory.

Unlike the other config how-tos, this one reads real OpenFOAM dictionaries
from the bundled cases, so it needs ``pybFoam`` (like the tutorials).
"""

# %%
# A helper to run LOAD + RESOLVE
# ------------------------------
# ``create_init`` builds the 3-stage initializer. We run LOAD (detect the
# algorithm + optional models, load configs) and RESOLVE (wire inter-model
# dependencies, e.g. boussinesq flipping ``use_boussinesq`` on PIMPLE), but
# stop before BUILD — collecting configs needs no mesh. The solver detects
# the algorithm from ``system/fvSolution`` relative to the working
# directory, so we ``chdir`` into the case for the load and restore the cwd
# afterwards (the configs are in memory by then).

import os
import shutil
import tempfile
from pathlib import Path

from neofoam import configurations
from neofoam.fields import load_fields, save_fields
from neofoam.framework.initialization import ConfigContext
from neofoam.io import save_configs
from neofoam.solver.incompressibleFluid import incompressibleFluid
from neofoam.solver.incompressibleFluid.create_fields import create_init
from neofoam.tutorial import clone_case

_ORIGINAL_CWD = Path.cwd()


# %%
# The full schema — without a case
# --------------------------------
# Before touching any case, ``configurations(incompressibleFluid)`` returns a
# case-free view of every config class the solver may consume: solver-core
# configs, the PIMPLE ``fvSchemes`` / ``fvSolution`` slices, and the configs
# of every registered optional model (here, boussinesq). This is the schema
# an agent fills — then saves — to build a case from scratch. The same call
# works for any solver, because the classes are declared on its spec.

schema = configurations(incompressibleFluid)

print("config classes:", schema.names)
for cls in schema:
    print(f"{cls.__name__:28s} fields: {sorted(cls.model_fields)}")


# %%
# Dict configs vs ``0/<name>`` field schemas
# ------------------------------------------
# ``configurations(...)`` surfaces both the dictionary configs
# (``constant/...``, ``system/...``) and the synthesised per-field
# schemas declared via ``Model.field(...)`` (``0/<name>`` files). The
# ``.dicts`` and ``.fields`` properties filter the view so callers can
# walk only one kind. The fields list reflects what the bound model
# specs declare — ``pimple`` owns ``0/U`` / ``0/p``, ``boussinesq``
# adds ``0/p_rgh`` / ``0/T`` / ``0/alphat``.

print("dict files:", sorted(cls.io_config.file for cls in schema.dicts))
print("0/* fields:", sorted(cls.io_config.file for cls in schema.fields))

# %%
# Look a class up, and read its JSON Schema
# -----------------------------------------
# The view is subscriptable by class name, and ``json_schema()`` returns the
# JSON Schema of every class — the form most agent / UI tooling consumes.

control_dict = schema["ControlDictConfig"]
print("ControlDictConfig fields:", sorted(control_dict.model_fields))
print("JSON schema keys        :", list(schema.json_schema()))


def load_and_resolve(case: Path):
    """Run LOAD + RESOLVE and return the resulting ``LoadResult``."""
    runner = create_init(case_dir=Path("."))
    os.chdir(case)
    try:
        load_result = runner.run_load()
        ctx = ConfigContext()
        for model in load_result.all_models:
            key = getattr(model, "name", None) or type(model).__name__.lower()
            ctx.register(key, model)
        runner.run_resolve(ctx)
    finally:
        os.chdir(_ORIGINAL_CWD)
    return load_result


# %%
# Collect from the pitzDaily case
# -------------------------------
# ``clone_case`` copies a bundled case to a throwaway tempdir. pitzDaily is
# a plain turbulent case — PIMPLE only, no optional models.

pitz = clone_case("pitzDaily")
pitz_result = load_and_resolve(pitz)

print("instances:", [type(c).__name__ for c in pitz_result.configs])
print("classes  :", [c.__name__ for c in pitz_result.config_classes])


# %%
# Print the loaded instances
# --------------------------
# Each config knows the file it came from (``file_name``) and, if it lives
# under a sub-dictionary, its ``subdict``. ``model_dump()`` is the parsed,
# validated content.

for cfg in pitz_result.configs:
    print(f"\n{type(cfg).__name__}  ->  {cfg.file_name}")
    if cfg.subdict:
        print(f"  subdict: {cfg.subdict}")
    print(f"  values : {cfg.model_dump()}")


# %%
# Print the declared schema set
# -----------------------------
# ``config_classes`` lists every config class the solver declares — not
# just the loaded ones. The PIMPLE ``fvSchemes`` / ``fvSolution`` slices
# appear here as *classes* even though their instances are not loaded (the
# OpenFOAM reader does not yet parse their typed scheme values). This is the
# schema an agent fills to scaffold a case.

for cls in pitz_result.config_classes:
    print(f"{cls.__name__:28s} fields: {sorted(cls.model_fields)}")


# %%
# Save the configs back to a fresh directory
# ------------------------------------------
# ``save_configs`` writes each instance to its registered
# path under ``case_dir``, reconstructing the case layout. Configs without
# an IO strategy are skipped with a warning rather than raising.

pitz_out = Path(tempfile.mkdtemp(prefix="neofoam_configs_pitz_"))
written = save_configs(pitz_result.configs, case_dir=pitz_out)

print("wrote:")
for path in written:
    print("  ", path.relative_to(pitz_out))


# %%
# Collect from the hotRoom case
# -----------------------------
# hotRoom is buoyancy-driven: its ``constant/transportProperties`` carries
# ``beta`` / ``TRef``, so the optional **boussinesq** model is detected at
# LOAD and flips ``use_boussinesq`` during RESOLVE. Its configs join the
# collection: ``BoussinesqConfig`` as an instance, plus the boussinesq
# ``fvSchemes`` / ``fvSolution`` slices in the schema set.

hot = clone_case("hotRoom")
hot_result = load_and_resolve(hot)

print("instances:", [type(c).__name__ for c in hot_result.configs])
print("classes  :", [c.__name__ for c in hot_result.config_classes])


# %%
# Saving hotRoom's configs
# ------------------------
# ``BoussinesqConfig`` has no IO strategy of its own — it co-owns
# ``constant/transportProperties`` with ``TransportPropertiesConfig`` — so
# ``save_configs`` skips it (with a warning) and writes it once via the
# transport config. We print its values so nothing is lost.

hot_out = Path(tempfile.mkdtemp(prefix="neofoam_configs_hot_"))
save_configs(hot_result.configs, case_dir=hot_out)

for cfg in hot_result.configs:
    if type(cfg).io_config is None:
        print(f"{type(cfg).__name__} (print-only): {cfg.model_dump()}")


# %%
# Read and write ``0/<name>`` fields with ``load_fields`` / ``save_fields``
# ------------------------------------------------------------------------
# Dict configs that share a target file (``constant/transportProperties``,
# ``system/fvSchemes``) need ``save_merged`` so multi-owner contributions
# survive. Field files are single-owner per ``0/<name>``, so
# :func:`neofoam.fields.load_fields` and :func:`neofoam.fields.save_fields`
# orchestrate the per-file read/write directly. The mapping returned by
# ``load_fields`` is keyed by field name (``"U"``, ``"p"``, …), so the
# caller can mutate one BC and save just that field — or save the whole
# set.

fields = load_fields(hot, solver=incompressibleFluid)
print("loaded fields:", sorted(fields))
print(
    "T@floor:",
    type(fields["T"].boundaryField["floor"]).__name__,
    fields["T"].boundaryField["floor"].model_dump(),
)

hot_fields_out = Path(tempfile.mkdtemp(prefix="neofoam_fields_hot_"))
written_fields = save_fields(fields, hot_fields_out)
print("wrote fields:")
for path in written_fields:
    print("  ", path.relative_to(hot_fields_out))


# %%
# Scaffold a case from configs alone
# ----------------------------------
# The reverse direction: build configs from values — no source case — via
# the schema view's ``new()`` (which pydantic-validates), then write them
# with ``save()``. An empty directory becomes a valid (partial) case. This
# is the building block for generating a case programmatically — an agent
# fills ``schema`` and saves it.

scaffold = Path(tempfile.mkdtemp(prefix="neofoam_scaffold_"))
schema.save(
    [
        schema.new(
            "ControlDictConfig",
            application="pimpleFoam",
            endTime=10.0,
            deltaT=0.01,
            writeInterval=1.0,
        ),
        schema.new("TransportPropertiesConfig", transportModel="Newtonian", nu=1e-5),
    ],
    case_dir=scaffold,
)

print("scaffolded:")
for path in sorted(scaffold.rglob("*")):
    if path.is_file():
        print("  ", path.relative_to(scaffold))

# Reload to prove the written files are valid.
reloaded = schema["ControlDictConfig"].load(case_dir=scaffold)
print("reloaded controlDict.endTime:", reloaded.endTime)


# %%
# See also
# --------
#
# - :doc:`/how-to/declare-fields` — how a model declares its ``0/<name>``
#   files (and the typed BC unions ``load_fields`` validates against).
# - :doc:`example_per_model_fvschemes` — how the per-model ``fvSchemes`` /
#   ``fvSolution`` slices in the schema set are declared.
# - :doc:`example_work_with_config_files` — loading and validating a single
#   config file.
"""Cleanup the temporary directories."""
for _tmp in (pitz, hot, pitz_out, hot_out, hot_fields_out, scaffold):
    shutil.rmtree(_tmp, ignore_errors=True)
