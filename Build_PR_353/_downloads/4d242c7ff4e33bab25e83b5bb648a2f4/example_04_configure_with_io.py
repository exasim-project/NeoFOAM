"""
Typed configs: YAML, JSON, and OpenFOAM dictionaries
====================================================

A *config* in NeoFOAM is a Pydantic ``BaseConfig`` subclass bound to a
file with the ``@IOStrategy(...)`` decorator. The same class can read
and write YAML, JSON, or an OpenFOAM dictionary — only the decorator
changes. Validation runs automatically on load, so a malformed file
fails fast with a clear error instead of crashing the solver mid-run.

This tutorial defines one config three ways, round-trips data through
each format, then shows the two subdict patterns that let several
configs share a single file: *flat* (``PIMPLE``) and *nested*
(``solvers.p`` or ``processing.stage1``). The OpenFOAM case used as
input lives under ``tutorials/configs_demo/`` in the repo — open it
alongside the script to see the file layout.

If you haven't run a NeoFOAM case yet, do
:doc:`example_01_run_incompressible_fluid` first.
"""

# %%
# Imports
# -------

from pydantic import BaseModel, Field, ValidationError

from neofoam.io import OF, JSON, YAML, BaseConfig, IOStrategy
from neofoam.tutorial import clone_case

# %%
# Clone the bundled case
# ----------------------
# ``configs_demo`` ships a YAML, a JSON, an OpenFOAM dictionary
# (``system/solverDict``), a shared-file example (``fvSolution.yaml``
# with both flat and nested subdicts), and a deeper OpenFOAM dict
# (``system/processing``) with two levels of nesting.

case = clone_case("configs_demo")
print(f"case: {case}")
for path in sorted(case.rglob("*")):
    if path.is_file():
        print(f"  {path.relative_to(case)}")

# %%
# Define one config — bind it to YAML
# -----------------------------------
# ``BaseConfig`` is a Pydantic model with two extras: ``.load()`` /
# ``.save()`` classmethods, and an ``io_config`` attached by
# ``@IOStrategy``. ``Field(gt=0, le=10000)`` etc. are standard Pydantic
# constraints — they're what powers the "fails fast on bad input"
# behaviour below.


@IOStrategy(YAML("solver_config.yaml"))
class SolverConfig(BaseConfig):
    tolerance: float = Field(gt=0)
    maxIterations: int = Field(gt=0, le=10000)
    solver: str


# %%
# Load
# ----
# ``load`` returns a typed instance — IDE autocomplete, mypy, and
# runtime validation all work the way they would on any Pydantic
# model. ``case_dir`` points at the cloned case; the file name comes
# from the ``@IOStrategy`` decorator.

cfg = SolverConfig.load(case_dir=case)
print(cfg)
print(f"  tolerance type: {type(cfg.tolerance).__name__}")

# %%
# Mutate and save
# ---------------
# A loaded config is a regular Pydantic model. Mutate fields and call
# ``.save()`` to round-trip back to the same file. The clone makes the
# write safe — the source under ``tutorials/configs_demo/`` is never
# touched.

cfg.tolerance = 1e-8
cfg.maxIterations = 500
cfg.save(case_dir=case)

print((case / "solver_config.yaml").read_text())

# %%
# Same data, different format: JSON
# ---------------------------------
# Swap ``YAML(...)`` for ``JSON(...)`` and nothing else changes — the
# field types, validation, ``.load()`` and ``.save()`` are identical.
# A real project picks one format per file and keeps it consistent;
# we use both here to show the decorator is the *only* coupling.


@IOStrategy(JSON("solver_config.json"))
class SolverConfigJSON(BaseConfig):
    tolerance: float = Field(gt=0)
    maxIterations: int = Field(gt=0, le=10000)
    solver: str


print(SolverConfigJSON.load(case_dir=case))

# %%
# Same data, OpenFOAM dictionary
# ------------------------------
# ``OF(...)`` uses ``pybFoam`` to parse the real OpenFOAM dictionary
# format — the same parser the solver uses, so what loads here is
# what the solver will see. Note the field names match the OpenFOAM
# keys verbatim; that's deliberate.


@IOStrategy(OF("system/solverDict"))
class SolverConfigOF(BaseConfig):
    tolerance: float = Field(gt=0)
    maxIterations: int = Field(gt=0, le=10000)
    solver: str


print(SolverConfigOF.load(case_dir=case))

# %%
# Flat subdict: multiple configs in one file
# ------------------------------------------
# OpenFOAM's ``system/fvSolution`` packs every linear solver, the
# PIMPLE outer-loop control, and tolerances into a single file. The
# ``subdict=...`` argument lets each config own one top-level section.


@IOStrategy(YAML("fvSolution.yaml", subdict="PIMPLE"))
class PIMPLEConfig(BaseConfig):
    nOuterCorrectors: int = Field(gt=0)
    nCorrectors: int = Field(gt=0)


pimple = PIMPLEConfig.load(case_dir=case)
print(f"PIMPLE: {pimple}")

# %%
# Nested subdict (dot path)
# -------------------------
# When a section lives several levels deep, pass a dot-separated path:
# ``subdict="solvers.p"`` selects the ``p`` block inside ``solvers``.
# Two configs reading the same file with different subdict paths is
# the canonical "share one fvSolution between many models" pattern.


@IOStrategy(YAML("fvSolution.yaml", subdict="solvers.p"))
class PSolverConfig(BaseConfig):
    solver: str
    tolerance: float = Field(gt=0)


@IOStrategy(YAML("fvSolution.yaml", subdict="solvers.U"))
class USolverConfig(BaseConfig):
    solver: str
    tolerance: float = Field(gt=0)


print(f"solvers.p: {PSolverConfig.load(case_dir=case)}")
print(f"solvers.U: {USolverConfig.load(case_dir=case)}")

# %%
# Nested subdict on OpenFOAM dictionaries
# ---------------------------------------
# The same dot-path syntax works against OpenFOAM dictionaries.
# ``system/processing`` (open the file in ``tutorials/configs_demo``
# to follow along) has a top-level ``metadata`` block plus a
# ``processing`` block that itself contains ``stage1`` and ``stage2``
# sub-blocks — two levels of nesting, mapped to two configs below.


@IOStrategy(OF("system/processing", subdict="metadata"))
class MetadataConfig(BaseConfig):
    name: str
    version: str
    priority: int = Field(gt=0)


@IOStrategy(OF("system/processing", subdict="processing.stage1"))
class Stage1Config(BaseConfig):
    algorithm: str
    threshold: float = Field(gt=0)
    maxIterations: int = Field(gt=0)
    batchSize: int = Field(gt=0)


@IOStrategy(OF("system/processing", subdict="processing.stage2"))
class Stage2Config(BaseConfig):
    algorithm: str
    threshold: float = Field(gt=0)
    maxIterations: int = Field(gt=0)
    batchSize: int = Field(gt=0)


print(f"metadata:           {MetadataConfig.load(case_dir=case)}")
print(f"processing.stage1:  {Stage1Config.load(case_dir=case)}")
print(f"processing.stage2:  {Stage2Config.load(case_dir=case)}")

# %%
# Structured field types: nested models, not just primitives
# ----------------------------------------------------------
# Config fields aren't restricted to ``int`` / ``str`` / ``float``.
# Any Pydantic model — including another ``BaseConfig`` — is a valid
# field type. The natural way to model the ``processing`` block in
# ``system/processing`` is one parent config whose ``stage1`` and
# ``stage2`` fields are themselves typed Pydantic models. Validation
# constraints (``Field(gt=0)``, etc.) on the nested type fire on
# load, exactly like for top-level fields.


class StageParams(BaseModel):
    algorithm: str
    threshold: float = Field(gt=0)
    maxIterations: int = Field(gt=0)
    batchSize: int = Field(gt=0)


@IOStrategy(OF("system/processing", subdict="processing"))
class ProcessingConfig(BaseConfig):
    stage1: StageParams
    stage2: StageParams


processing = ProcessingConfig.load(case_dir=case)
print(f"stage1 type:      {type(processing.stage1).__name__}")
print(f"stage1.algorithm: {processing.stage1.algorithm}")
print(f"stage2.threshold: {processing.stage2.threshold}")
print("JSON schema knows about the nesting:")
print(f"  {list(ProcessingConfig.model_json_schema()['$defs'].keys())}")

# %%
# Why bother with a nested model?
# -------------------------------
# Same data, two ways:
#
# - **Three flat configs** (``Stage1Config``, ``Stage2Config``,
#   ``MetadataConfig`` above) — one per subdict path. Each is its own
#   ``@spec.config`` target.
# - **One structured config** (``ProcessingConfig``) with nested
#   Pydantic fields. The class hierarchy mirrors the file hierarchy,
#   the JSON schema records the nesting, and downstream code can pass
#   ``processing.stage1`` around as a typed object instead of looking
#   up a flat dict by name.
#
# Pick the flat form when each subdict is owned by a different model
# (the ``solvers.p`` / ``solvers.U`` pattern further up); pick the
# nested form when one model owns the whole block (an algorithm that
# uses both stages together).

# %%
# Partial save preserves the rest of the file
# -------------------------------------------
# When a config uses a subdict, ``.save()`` only rewrites that
# section. The ``stage2`` entry below is untouched even though we
# only edited ``stage1``.

stage1 = Stage1Config.load(case_dir=case)
stage1.threshold = 1e-5
stage1.maxIterations = 250
stage1.save(case_dir=case)

print((case / "system" / "processing").read_text())

# %%
# Validation: bad data fails on load
# ----------------------------------
# ``Field(gt=0)`` rejects ``tolerance: -1`` in
# ``configs_demo/bad_solver.yaml``. By default ``load()`` raises
# ``ValidationError``; catch it to surface the message in a UI or
# test.

try:
    SolverConfig.load(case_dir=case, file="bad_solver.yaml")
except ValidationError as exc:
    print("Caught ValidationError:")
    for err in exc.errors():
        print(f"  {err['loc']}: {err['msg']}")

# %%
# Collect errors instead of raising
# ---------------------------------
# ``collect_errors`` is the non-raising variant — returns an empty
# list on success, a list of ``ValidationErrors`` (with file name and
# subdict context) on failure. Useful in CI and in editors that want
# to surface every problem at once rather than the first one.

errors = SolverConfig.collect_errors(case_dir=case, file="bad_solver.yaml")
for e in errors:
    print(f"{e.file_name}: {e.field} — {e.message}")

# %%
# Where to go from here
# ---------------------
# - :doc:`example_02_passive_scalar_plugin` shows a config attached
#   to a plugin model via ``@spec.load``.
