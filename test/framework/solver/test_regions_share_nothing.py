# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

# NB: no ``from __future__ import annotations`` — the dependency resolver matches
# ``param.annotation is Context``.

"""Two runtimes of one solver spec share no run state, as two regions run them.

``shared`` records the value of a cached ``Depends`` (the label of the context)
in its one operation. Each run instantiates the spec, initializes and runs it
on a thread of its own, one after the other: a cache on the spec would hand
the second run the first run's value.
"""

import threading
from typing import Annotated, Any

from neofoam.framework.context import Context
from neofoam.framework.graph import DAGResolver
from neofoam.framework.initialization.depends import Depends
from neofoam.framework.operations import Operations, StepBuilder
from neofoam.framework.solver import Solver


def label_of(ctx: Context) -> Any:
    return ctx.models["label"]


shared = Solver("shared_by_regions")


@shared.initializer
def _initialize(self: Any) -> Context:
    return Context(fields={}, models={"seen": []})


@shared.execution_graph_step
def _execution_graph(self: Any, ctx: Context) -> tuple[StepBuilder, Operations]:
    builder = StepBuilder()
    builder.step(self.operations["record"])
    return builder, Operations()


@shared.operation()
def record(self: Any, ctx: Context, region: Annotated[Any, Depends(label_of)]) -> None:
    ctx.models["seen"].append(region)


def _run(label: str, contexts: dict[str, Context]) -> None:
    runtime = shared.instantiate()
    ctx = runtime.initialize()
    ctx.models["label"] = label
    builder, model_ops = runtime.execution_graph(ctx=ctx)
    DAGResolver().resolve(builder, model_ops).operations.run(ctx)
    contexts[label] = ctx


def test_two_runtimes_of_one_solver_spec_resolve_their_own_cached_dependency() -> None:
    contexts: dict[str, Context] = {}
    for region in ("left", "right"):
        thread = threading.Thread(target=_run, args=(region, contexts))
        thread.start()
        thread.join()

    assert {region: ctx.models["seen"] for region, ctx in contexts.items()} == {
        "left": ["left"],
        "right": ["right"],
    }
