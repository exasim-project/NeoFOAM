"""
Add config files to a solver and model
======================================

A solver and the models it runs each carry their own
:class:`~neofoam.io.BaseConfig`. The framework reads them from disk at
LOAD time and threads them into operations by type annotation, so the
operation code never reaches into ``ctx`` for its own parameters.

User-facing API — five shapes
-----------------------------

Every config goes through the spec (model or solver). Five forms cover
everything:

1. **Single config — class form**: ``spec.config(SqrtSolverConfig)``.
   The class is self-describing via ``@IOStrategy(...)``; the framework
   auto-locates the instance after load.
2. **Single config — callback form (escape hatch)**:
   ``@spec.config`` over a ``def _load(case_dir: Path) -> SqrtSolverConfig``.
   Use when load needs custom args (``validate=False``, defaults
   overrides) or merges from multiple sources.
3. **Multiple configs**: call ``spec.config(Cls)`` repeatedly.
   ``runtime.config`` then becomes a ``SimpleNamespace`` keyed by
   snake-case class name; type-injection still works in operations.
4. **fvSchemes / fvSolution slices**: ``Sub = spec.config(fvSchemes)``
   returns a per-spec subclass; operations extend it via
   ``@Sub.add(div="div(phi,U)")`` — typed Pydantic fields are injected
   incrementally. See :doc:`example_per_model_fvschemes` for the full
   pattern.
5. **Consume in operations / lifecycle callbacks**: annotate the
   parameter with the config class. The framework finds it on
   ``runtime.config`` by type and passes it in. Operations never read
   configs from ``ctx``.

External validation in one call::

    errors = runner.run_load().validate()      # Pydantic over every loaded config

The rest of this page builds the smallest end-to-end example that
exercises shapes 1, 2 (callback as fall-back), and 5 — a Babylonian
square-root iteration ``x_{n+1} = 0.5 * (x_n + target / x_n)`` — and
asserts the result matches :func:`math.sqrt`. The arithmetic is
incidental; configs are the subject.
"""

# %%
# Stage the YAML fixtures into a case directory
# ---------------------------------------------
# Every :class:`BaseConfig` binds to a *filename*, not a full path.
# The loader resolves that filename against the ``case_dir`` you pass
# in. Here we copy the two YAML fixtures shipped alongside this page
# into a throwaway directory and use it as the case.

import inspect
import math
import shutil
import tempfile
from pathlib import Path
from typing import Annotated, Any, Optional

from pydantic import Field

from neofoam.framework.context import Context, FieldUpdates
from neofoam.framework.graph import DAGResolver
from neofoam.framework.initialization import (
    Depends,
    InitializerBuilder,
    InitStep,
    LoadResult,
    StagedInitRunner,
    StagedInitSpec,
)
from neofoam.framework.model import Model
from neofoam.framework.operations import (
    IterativeOp,
    Operation,
    Operations,
    StepBuilder,
)
from neofoam.framework.solver import Solver
from neofoam.framework.types import OperationMetadata
from neofoam.io import BaseConfig, IOStrategy, YAML


def _here() -> None:
    """Marker used to locate this script on disk via inspect.getfile()."""


HERE = Path(inspect.getfile(_here)).resolve().parent
CASE_DIR = Path(tempfile.mkdtemp(prefix="neofoam_sqrt_"))
shutil.copy(HERE / "babylonian_config.yaml", CASE_DIR / "babylonian_config.yaml")
shutil.copy(HERE / "sqrt_solver_config.yaml", CASE_DIR / "sqrt_solver_config.yaml")
print("case_dir =", CASE_DIR)


# %%
# Declare the model's config
# --------------------------
# The model owns its problem statement: the value whose square root
# we want and the seed for the iteration. ``@IOStrategy(YAML(...))``
# binds the class to a file inside ``case_dir``; the Pydantic
# ``Field(...)`` validators reject bad input on load.
#
# .. literalinclude:: babylonian_config.yaml
#    :language: yaml


@IOStrategy(YAML("babylonian_config.yaml"))
class BabylonianConfig(BaseConfig):
    target: float = Field(gt=0, description="value whose square root is wanted")
    initial_guess: float = Field(gt=0, default=1.0)


# %%
# Declare the solver's config
# ---------------------------
# The solver owns the *algorithm controls* — when to stop and how
# many iterations to allow. Keeping these on a separate config makes
# each layer responsible for its own knobs.
#
# .. literalinclude:: sqrt_solver_config.yaml
#    :language: yaml


@IOStrategy(YAML("sqrt_solver_config.yaml"))
class SqrtSolverConfig(BaseConfig):
    max_iterations: int = Field(gt=0, default=50)
    tolerance: float = Field(gt=0, default=1e-12)


# %%
# Define the model: register, build, operation
# --------------------------------------------
# Declared at module scope, the :class:`Model` earns three calls:
#
# - ``babylonian.config(BabylonianConfig)`` (shape 1 — class form)
#   registers the config class. The framework auto-loads it via the
#   class's ``@IOStrategy`` binding on ``babylonian.instantiate(case_dir=…)``,
#   so no callback is needed. Use the callback form
#   ``@babylonian.load`` only when load needs custom args
#   (``validate=False``, multi-source merge) or manifest-driven
#   multi-instance loading.
# - ``@babylonian.build`` emits the :class:`InitStep` objects that
#   produce the fields the operation will read. ``cfg`` is
#   auto-injected from ``runtime.config`` because its annotation
#   matches (shape 5).
# - ``@babylonian.operation`` is the work itself. ``x: float`` is
#   looked up in ``ctx.fields`` by *name*; ``cfg: BabylonianConfig``
#   is auto-injected from ``runtime.config`` by *type* (shape 5).

babylonian = Model("Babylonian")
babylonian.config(BabylonianConfig)


@babylonian.build
def _bab_build(cfg: BabylonianConfig) -> list[InitStep]:
    return [
        InitStep(
            name="x",
            initializer=lambda _ctx: cfg.initial_guess,
            depends_on=[],
            category="fields",
        ),
    ]


@babylonian.operation(operation_number="1.0")
def babylonian_step(self: Any, x: float, cfg: BabylonianConfig) -> FieldUpdates:
    """One Babylonian iteration: x ← 0.5 * (x + target / x)."""
    return FieldUpdates({"x": 0.5 * (x + cfg.target / x)})


# %%
# Wire the solver via StagedInitSpec
# ----------------------------------
# Solvers don't host the build stage directly — a
# :class:`StagedInitSpec` registers LOAD and BUILD (and an optional
# RESOLVE, omitted here) as independent callbacks.
#
# - **LOAD** returns the solver config plus every model runtime in a
#   single ``LoadResult``. The model runtime carries its own config
#   (auto-loaded via ``babylonian.config(...)``).
# - **BUILD** receives the model runtimes and collects their
#   ``InitStep`` lists. Solver config does *not* go through
#   ``InitializerBuilder.add_*`` — the framework picks the
#   ``SqrtSolverConfig`` instance out of ``LoadResult.core_models``
#   and assigns it to ``runtime.config`` because the spec registered
#   ``SqrtSolverConfig`` via ``@spec.config`` (next section).
#
# BUILD's signature is inspected: declare only the kwargs you need.
# Here we ask for ``optional_models`` (the non-config-class entries
# from LOAD); ``core_models`` and ``models`` are the other accepted
# names.


def create_init(case_dir: Optional[Path] = None) -> StagedInitRunner:
    spec_builder = StagedInitSpec.build("SqrtSolver")
    resolved_case_dir = case_dir or CASE_DIR

    @spec_builder.load
    def _load() -> LoadResult:
        solver_cfg = SqrtSolverConfig.load(case_dir=resolved_case_dir, validate=False)
        bab = babylonian.instantiate(case_dir=resolved_case_dir)
        return LoadResult(core_models=[solver_cfg], optional_models=[bab])

    @spec_builder.build
    def _build(core_models: list[Any], optional_models: list[Any]) -> list[InitStep]:
        builder = InitializerBuilder()
        for runtime in optional_models:
            builder.extend(runtime.run_build())
        return builder.build()

    return StagedInitRunner(spec_builder.finalize())


# %%
# Define the solver: register its config + execution graph
# --------------------------------------------------------
# The solver registers ``SqrtSolverConfig`` against its spec via
# ``spec.config(SqrtSolverConfig)`` — **shape 1, the class form**.
# The framework locates the matching instance in ``core_models``
# during ``initialize()`` and assigns it to ``runtime.config``;
# operations and lifecycle callbacks then receive it by type
# (shape 5), the same way model operations do.
#
# To register multiple configs against the same spec
# (e.g. ``SqrtSolverConfig`` *plus* a separate ``DiagnosticsConfig``),
# call ``.config(...)`` again — ``runtime.config`` becomes a
# ``SimpleNamespace`` keyed by snake-case class name (shape 3) and
# every type-injected parameter still finds its instance by class.
#
# To override the auto-load (custom ``validate`` flag, multi-source
# merge), use the **callback form** — ``@sqrt_solver_spec.config``
# over ``def _load(case_dir: Path) -> SqrtSolverConfig`` (shape 2).
#
# The execution graph is a single iterative loop. The convergence
# check is constructed inside ``_execution_graph`` with the loaded
# config and target value baked in as a closure; its ``__call__(ctx)``
# only ever reads ``ctx.fields`` — no ``ctx.models`` access required.

sqrt_solver_spec = Solver("SqrtSolver")
sqrt_solver_spec.config(SqrtSolverConfig)


class ConvergenceCheck:
    def __init__(self, cfg: SqrtSolverConfig, target: float) -> None:
        self._cfg = cfg
        self._target = target
        self._iteration = 0

    def __call__(self, ctx: Context) -> bool:
        self._iteration += 1
        if self._iteration > self._cfg.max_iterations:
            return False
        x = ctx.fields["x"]
        return abs(x * x - self._target) > self._cfg.tolerance


@sqrt_solver_spec.initializer
def _initialize(
    self: Any, init: Annotated[StagedInitRunner, Depends(create_init)]
) -> Context:
    return init.run()


@sqrt_solver_spec.execution_graph_step
def _execution_graph(
    self: Any,
    cfg: SqrtSolverConfig,
    domain_name: Optional[str] = None,
) -> tuple[StepBuilder, Operations]:
    _ = domain_name
    target = next(
        rt.config.target
        for rt in self.state.optional_models
        if isinstance(rt.config, BabylonianConfig)
    )
    builder = StepBuilder()
    loop = Operation(
        func=IterativeOp(ConvergenceCheck(cfg, target)),
        metadata=OperationMetadata(op_name="sqrt_loop"),
    )
    loop_builder = builder.loop(loop)
    for model_runtime in self.state.optional_models:
        for op in model_runtime.operations:
            loop_builder.step(op)
    return builder, Operations()


# %%
# Validate every loaded config without running the solver
# -------------------------------------------------------
# Useful for CLI / GUI front-ends that want to surface every config
# the solver and its models declared, and re-validate them before
# BUILD ever runs. ``runner.run_load()`` executes the LOAD stage in
# isolation; ``.configs`` returns a flat list and ``.validate()``
# runs ``neofoam.io.validate_models`` over it.

_runner = create_init()
_load_result = _runner.run_load()
print("Loaded configs:")
for _cfg in _load_result.configs:
    print(" ", type(_cfg).__name__, "→", _cfg.model_dump())

_errors = _load_result.validate()
assert _errors == [], _errors
print(f"validation errors: {len(_errors)}")


# %%
# Run it
# ------
# Instantiate the spec, call ``.initialize()`` (which executes the
# three-stage init), build the execution graph, resolve it into a
# flat operation list, and run. After the loop exits,
# ``ctx.fields["x"]`` carries the converged answer.

sqrt_solver = sqrt_solver_spec.instantiate()
ctx = sqrt_solver.initialize()
builder, model_ops = sqrt_solver.execution_graph()
resolved = DAGResolver().resolve(builder, model_ops)
resolved.operations.run(ctx)

target = BabylonianConfig.load(case_dir=CASE_DIR).target
print(f"target          = {target}")
print(f"computed sqrt   = {ctx.fields['x']!r}")
print(f"math.sqrt       = {math.sqrt(target)!r}")
print(f"|difference|    = {abs(ctx.fields['x'] - math.sqrt(target)):.2e}")
assert abs(ctx.fields["x"] - math.sqrt(target)) < 1e-6


# %%
# See also
# --------
#
# - :doc:`example_per_model_fvschemes` — shape 4: per-model
#   ``fvSchemes`` / ``fvSolution`` subclasses built up by
#   ``@<Sub>.add(...)`` decorators on each operation.
# - :doc:`example_work_with_config_files` — declare, load, validate,
#   and subdict-isolate :class:`BaseConfig` classes on their own.
# - :doc:`example_register_a_model` — the model side in isolation,
#   including ``@spec.resolve`` and ``@spec.detect``.
# - :doc:`example_use_depends_for_injection` — pull values from
#   ``ctx.models`` or external providers via ``Annotated[T, ...]``.
# - :doc:`/explanation/three-stage-init` — why LOAD / RESOLVE / BUILD
#   are split.
