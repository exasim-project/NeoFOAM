"""
Add config files to a solver and model
======================================

**Key question.** How do I attach configuration to the things that
run — a *model*, a *solver*, and the per-spec *fvSchemes / fvSolution*
slices — and read it inside an operation without reaching into a
global ``ctx``?

**Answer.** Each config is owned by the spec that declares it:

.. code-block:: python

    spec.config(MyConfig)                 # declare: this spec owns MyConfig

Operations receive it by type annotation:

.. code-block:: python

    def op(self, x, cfg: MyConfig):       # consume: by type, not via ctx
        ...                               # framework injects the loaded instance

The same mechanic applies on a ``Model``, on a ``Solver``, and (with a
typed-slice twist) on ``fvSchemes`` / ``fvSolution``. The injector
itself is shared code in :mod:`neofoam.framework.config_injection`:
config parameters resolve from ``runtime.config`` by *type*; field
parameters resolve from ``ctx.fields`` by *name*.

The rest of this page proves each surface with the smallest runnable
example.
"""

# %%
# Stage the YAML fixtures
# -----------------------
# Each ``BaseConfig`` binds to a *filename*; the loader resolves it
# against ``case_dir``. The three fixtures shipped next to this page
# are staged into a throwaway directory.
#
# .. literalinclude:: ../../examples/how-to/babylonian_config.yaml
#    :language: yaml
#    :caption: babylonian_config.yaml
# .. literalinclude:: ../../examples/how-to/sqrt_solver_config.yaml
#    :language: yaml
#    :caption: sqrt_solver_config.yaml
# .. literalinclude:: ../../examples/how-to/per_model_fvSchemes.yaml
#    :language: yaml
#    :caption: per_model_fvSchemes.yaml

import inspect
import shutil
import tempfile
from pathlib import Path
from typing import Any

from pydantic import Field

from neofoam.foam import fvSchemes
from neofoam.framework.context import Context, FieldUpdates
from neofoam.framework.model import Model
from neofoam.framework.solver import Solver
from neofoam.io import BaseConfig, IOStrategy, YAML


def _here() -> None:
    """Marker used to locate this script on disk via inspect.getfile()."""


HERE = Path(inspect.getfile(_here)).resolve().parent
CASE = Path(tempfile.mkdtemp(prefix="neofoam_cfg_"))
for _name in (
    "babylonian_config.yaml",
    "sqrt_solver_config.yaml",
    "per_model_fvSchemes.yaml",
):
    shutil.copy(HERE / _name, CASE / _name)


# %%
# Model — declare + consume
# -------------------------
# ``model.config(ProblemConfig)`` is the declaration. The operation
# names the config in its signature; ``rt.operations[0].run(ctx)`` is
# the consumption: ``cfg`` is injected from ``rt.config`` by type,
# ``x`` is pulled from ``ctx.fields`` by name. The operation body
# touches ``cfg.target`` directly — never ``ctx`` — and that is the
# whole point.


@IOStrategy(YAML("babylonian_config.yaml"))
class ProblemConfig(BaseConfig):
    target: float = Field(gt=0)
    initial_guess: float = Field(gt=0, default=1.0)


newton = Model("Newton")
newton.config(ProblemConfig)  # declare


@newton.operation(operation_number="1.0")
def step(self: Any, x: float, cfg: ProblemConfig) -> FieldUpdates:
    return FieldUpdates({"x": 0.5 * (x + cfg.target / x)})  # consume


rt = newton.instantiate(case_dir=CASE)
assert isinstance(rt.config, ProblemConfig)  # declare → autoloaded
ctx = Context(fields={"x": rt.config.initial_guess}, models={})
rt.operations[0].run(ctx)  # consume → injected
print("model: x after one step =", ctx.fields["x"])
assert ctx.fields["x"] == 0.5 * (1.0 + 2.0 / 1.0)


# %%
# Solver — same declaration, same injection
# -----------------------------------------
# ``solver.config(ControlsConfig)`` is the identical one-liner.
# A solver populates ``runtime.config`` during ``initialize()``: the
# framework pulls every registered config class out of
# ``state.core_models`` and assigns it to ``runtime.config``. After
# that, operations consume the config by type using the *same*
# shared injector as the model — no separate mechanism.
#
# Everything else a real solver carries (StagedInitSpec, execution
# graphs, DAG resolution) is plumbing for *running* an algorithm, not
# for *attaching configs*. It is covered in
# :doc:`/auto_tutorials/example_03_build_a_solver`.


@IOStrategy(YAML("sqrt_solver_config.yaml"))
class ControlsConfig(BaseConfig):
    max_iterations: int = Field(gt=0, default=50)
    tolerance: float = Field(gt=0, default=1e-12)


sqrt = Solver("Sqrt")
sqrt.config(ControlsConfig)  # declare — same call as on the model


@sqrt.initializer
def _init(self: Any) -> Context:
    # Minimal initializer: load the config into state.core_models so
    # the framework can pick it up onto runtime.config.
    self.state.core_models = [ControlsConfig.load(case_dir=CASE)]
    return Context(fields={"x": 9.0}, models={})


@sqrt.operation(operation_number="1.0")
def report(self: Any, x: float, cfg: ControlsConfig) -> None:
    print(f"solver: tolerance={cfg.tolerance}, x={x}")  # consume — by type


srt = sqrt.instantiate()
sctx = srt.initialize()
assert isinstance(srt.config, ControlsConfig)  # declare → on runtime.config
for op in srt.operations:
    op.run(sctx)  # consume → injected


# %%
# fvSchemes / fvSolution — declare a typed slice, add entries, load
# -----------------------------------------------------------------
# A model that needs an entry from ``system/fvSchemes`` declares a
# *per-spec subclass* with the same one-liner: ``spec.config(fvSchemes)``
# returns a fresh class. Operations then extend it with typed entries
# via ``@<Sub>.add(...)``. Loading the subclass yields a Pydantic
# instance whose values have been typechecked against the OpenFOAM
# scheme unions in :mod:`neofoam.foam.schemes`.
#
# The full story (multiple sections, ``fvSolution`` too, bad-input
# behaviour, independent subclasses per spec) lives in
# :doc:`example_per_model_fvschemes`.

pimple = Model("Pimple")
PimpleFvSchemes = pimple.config(fvSchemes)  # declare — returns a subclass
IOStrategy(YAML("per_model_fvSchemes.yaml"))(PimpleFvSchemes)


@pimple.operation(operation_number="2.1")
@PimpleFvSchemes.add(div="div(phi,U)")  # consume — typed entry on the slice
def momentum() -> None:
    """Pretend U-momentum operation."""


schemes = PimpleFvSchemes.load(case_dir=CASE)
print("fvSchemes: div(phi,U) =", schemes.divSchemes.div_phi_U)


# %%
# See also
# --------
#
# - :doc:`example_work_with_config_files` — declaring, loading,
#   validating, and subdict-isolating ``BaseConfig`` classes on their
#   own (single-config focus).
# - :doc:`example_per_model_fvschemes` — the typed-slice mechanic in
#   full, including per-operation entry declarations and bad-input
#   behaviour.
# - :doc:`example_collect_and_save_configs` — case-free schema of a
#   whole solver via ``configurations(spec)``, and scaffolding a case
#   from configs alone.
# - :doc:`example_register_a_model` — the model side end-to-end
#   (``@spec.resolve``, ``@spec.detect``).
# - :doc:`/explanation/three-stage-init` — why LOAD / RESOLVE / BUILD
#   are split.
