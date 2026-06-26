# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

"""Populate a :class:`~neofoam.framework.context.Context` before any
operation runs.

NeoFOAM solvers can't start their main loop until fields, operators,
models, and the mesh are constructed and wired together. This
package owns that pre-run work and splits it into three layers:

- **Steps & helpers** — the unit of deferred initialization is an
  :class:`~neofoam.framework.initialization.init_step.InitStep`: a
  name, its dependencies, and a factory callable. The four helpers
  (:func:`field`, :func:`operator`, :func:`model`, :func:`lazy`)
  apply the canonical name prefix and category;
  :class:`InitializerBuilder` collects them via a fluent API.
- **Execution** — :func:`execute_initialization` validates the step
  graph, sorts topologically, runs each step in order, and routes
  the results into the Context. See
  :mod:`~neofoam.framework.initialization.execution`.
- **Three-stage pipeline** — for solvers that need to LOAD →
  RESOLVE → BUILD, :class:`StagedInitSpec` (decorator-driven
  registration) plus :class:`StagedInitRunner` (execution)
  orchestrate the three stages and hand off to
  :func:`execute_initialization` at the end. See
  :mod:`~neofoam.framework.initialization.staged`.

Dependency injection of resolved values into solver callbacks is
expressed with :class:`Depends`; routing is open for extension via
:class:`CategoryRouter` (register a custom handler, pass it to
:func:`execute_initialization`).

See :doc:`/explanation/three-stage-init` for the design rationale
and stage semantics.
"""

from .config_context import ConfigContext
from .init_step import InitCategory, InitStep, InitStepExecutionError
from .helpers import field, operator, lazy, model, InitializerBuilder
from .execution import (
    CategoryRouter,
    InitResult,
    InitializationGraphError,
    execute_initialization,
    execute_step,
)
from .depends import Depends
from .staged import (
    LoadResult,
    StagedInitRunner,
    StagedInitSpec,
    StagedInitSpecBuilder,
)

__all__ = [
    "ConfigContext",
    "InitStep",
    "InitCategory",
    "InitStepExecutionError",
    "InitResult",
    "InitializationGraphError",
    "CategoryRouter",
    "field",
    "operator",
    "lazy",
    "model",
    "InitializerBuilder",
    "execute_initialization",
    "execute_step",
    "Depends",
    "LoadResult",
    "StagedInitRunner",
    "StagedInitSpec",
    "StagedInitSpecBuilder",
]
