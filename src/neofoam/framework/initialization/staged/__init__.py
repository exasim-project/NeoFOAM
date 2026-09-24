# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""3-stage initialization framework: spec + runner.

A solver's pre-run setup is split into three explicit stages —
**LOAD**, **RESOLVE**, **BUILD** — registered as decorator-driven
callbacks on a :class:`StagedInitSpec` and executed by a
:class:`StagedInitRunner`:

- **LOAD** reads configuration files and instantiates model objects
  in isolation. Returns a :class:`LoadResult` carrying the core and
  optional models.
- **RESOLVE** wires inter-model dependencies through a
  :class:`~neofoam.framework.initialization.config_context.ConfigContext`
  so models can adapt their configuration to their peers.
- **BUILD** produces a list of
  :class:`~neofoam.framework.initialization.init_step.InitStep`
  objects describing how to construct runtime objects (fields,
  operators, models). The framework topologically sorts and runs
  them via
  :func:`~neofoam.framework.initialization.execution.execute_initialization`.

The split separates the **immutable description** of a pipeline
(:class:`StagedInitSpec`, populated by
:class:`StagedInitSpecBuilder`) from the **mutable execution state**
that the run produces (:class:`StagedInitRunner`'s ``argv``,
``core_models``, ``optional_models``, ``state``). The spec is
hashable, reusable, and safe to share; the runner is per-instance.

See :doc:`/explanation/three-stage-init` for the stage semantics and
how the pipeline fits into the solver lifecycle.
"""

from .runner import StagedInitRunner
from .spec import LoadResult, StagedInitSpec, StagedInitSpecBuilder

__all__ = [
    "LoadResult",
    "StagedInitRunner",
    "StagedInitSpec",
    "StagedInitSpecBuilder",
]
