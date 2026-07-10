"""
Register a new model
====================

Declare a :class:`~neofoam.framework.model.spec.ModelSpec` at module
import time, attach stage callbacks with decorators, and let a solver
detect it via the plugin system or load it directly.
"""

# %%
# Minimum spec
# ------------
# A model needs at least a ``@spec.load`` — it returns the per-instance
# config the runtime will carry. That alone makes
# ``turbulence.instantiate(case_dir, instance_id)`` work; the resulting
# :class:`ModelRuntime` exposes ``.config`` plus an empty operations
# list.

from pathlib import Path
from typing import Any

from neofoam.framework.context import FieldUpdates
from neofoam.framework.initialization import (
    ConfigContext,
    InitializerBuilder,
    InitStep,
)
from neofoam.framework.initialization.staged import LoadResult, StagedInitSpec
from neofoam.framework.model import Model
from neofoam.framework.solver import Solver
from neofoam.io import IOStrategy, YAML, BaseConfig


@IOStrategy(YAML("turbulence_config.yaml"))
class TurbulenceConfig(BaseConfig):
    nut: float = 1e-3


turbulence = Model("Turbulence")


@turbulence.load
def _load(case_dir: Path, instance_id: str) -> TurbulenceConfig:
    return TurbulenceConfig.load(case_dir=case_dir, validate=True)


# %%
# Add operations
# --------------
# Use ``@spec.operation(...)`` to contribute to the solver's DAG. The
# first positional ``self`` is bound to the runtime; ``BaseConfig``
# parameters are injected from ``runtime.config`` by type; plain-typed
# parameters (``float``, ``int``, …) are looked up in ``ctx.fields``
# by name.


@turbulence.operation(operation_number="2.5", depends_on=["fields.U"])
def update_nut(self: Any, U: float, cfg: TurbulenceConfig) -> FieldUpdates:
    return FieldUpdates({"nut": cfg.nut * abs(U)})


# %%
# Optional stages
# ---------------
# Three more decorators expand the spec when needed:
#
# - ``@spec.resolve(config, ctx)`` — adjust ``config`` after every
#   model has loaded but before any field is constructed. Use this to
#   read cross-model decisions from the :class:`ConfigContext`.
# - ``@spec.build(config)`` — return ``list[InitStep]`` to produce
#   fields/operators during the BUILD stage. The runner adds them to
#   the global init graph.
# - ``@spec.detect()`` — return ``True`` if the model should be loaded
#   for this case. Used by plugin discovery; defaults to ``True``.


@turbulence.resolve
def _resolve(config: TurbulenceConfig, ctx: ConfigContext) -> None:
    # Read sibling configs from ctx, mutate this one if needed.
    pass


@turbulence.build
def _build(config: TurbulenceConfig) -> list[InitStep]:
    # Emit init steps for fields/operators owned by this model.
    return []


# %%
# Plug into a solver via StagedInit
# ---------------------------------
# Solvers don't host the build stage directly — they delegate to a
# :class:`StagedInitSpec` that registers LOAD / RESOLVE / BUILD as
# three independent callbacks. The build stage receives the core and
# optional models the LOAD stage produced, hands them to an
# :class:`InitializerBuilder`, and returns the flat list of
# :class:`InitStep` objects the runner will execute.
# ``add_core_models`` / ``add_optional_models`` accept anything
# matching the :class:`BuildsInitSteps` Protocol (one method:
# ``run_build() -> list[InitStep]``) — every ``ModelRuntime`` already
# qualifies.

solver_spec = Solver("demo_solver")

init = StagedInitSpec.build("demo_solver")


@init.load
def _load() -> LoadResult:
    return LoadResult(core_models=[], optional_models=[])


@init.build
def _solver_build(core_models: list[Any], optional_models: list[Any]) -> list[InitStep]:
    builder = InitializerBuilder()
    builder.add_core_models(core_models)
    builder.add_optional_models(optional_models)
    return builder.build()


# %%
# Declaring on-disk fields
# ------------------------
# ``Model.config(...)`` registers the dictionary configs (``constant/...``
# / ``system/...``) the model owns. The parallel for ``0/<name>`` field
# files is ``Model.field(name, dimensions=..., value_type=..., allowed_bcs=
# [...])``. The handle it returns cross-links to ``@spec.build`` via
# ``decl.create(read_fn)``, and the synthesised per-field schema surfaces
# in ``configurations(solver).fields`` so a case author / agent fills
# the BCs without copying a source case. See
# :doc:`/how-to/declare-fields` for the full surface.


# %%
# See also
# --------
#
# - :doc:`/how-to/declare-fields` — register a model's ``0/<name>`` files
#   with typed BC unions.
# - :doc:`/explanation/model-structure` — why the ``Spec`` / ``Runtime``
#   split exists.
# - :doc:`/reference/model/index` — module API reference.
