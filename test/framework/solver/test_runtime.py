# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

# NB: no ``from __future__ import annotations`` — the dependency resolver matches
# ``param.annotation is Context``.

"""Two regions that run one solver spec share no run state, as multiRun runs them.

``shared`` records the value of a cached ``Depends`` (the region of the context)
in its one operation. Each region instantiates the spec, initializes and runs
it on a thread of its own, one after the other: a cache on the spec would hand
the second region the first region's value.
"""

import threading
from typing import Annotated, Any, Optional

import pytest

from framework.solver.region_counter import counter
from neofoam.framework.context import Context
from neofoam.framework.initialization.depends import Depends
from neofoam.framework.operations import Operations, StepBuilder
from neofoam.framework.solver import Solver


def region_of(ctx: Context) -> Any:
    return ctx.region


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
def record(self: Any, ctx: Context, region: Annotated[Any, Depends(region_of)]) -> None:
    ctx.models["seen"].append(region)


def _run(region: str, contexts: dict[str, Context]) -> None:
    runtime = shared.instantiate(region=region)
    ctx = runtime.initialize()
    runtime.resolve_operations(ctx).run(ctx)
    contexts[region] = ctx


def test_two_regions_of_one_solver_spec_resolve_their_own_cached_dependency() -> None:
    contexts: dict[str, Context] = {}
    for region in ("left", "right"):
        thread = threading.Thread(target=_run, args=(region, contexts))
        thread.start()
        thread.join()

    assert {region: ctx.models["seen"] for region, ctx in contexts.items()} == {
        "left": ["left"],
        "right": ["right"],
    }


@pytest.mark.parametrize(("region", "expected"), [("left", 2.0), ("right", 3.0), (None, 1.0)])
def test_runtime_region_selects_the_region_configs(region: Optional[str], expected: float) -> None:
    ctx = counter.instantiate(region=region).initialize()

    assert ctx.models["increment"] == expected


def test_runtime_initialize_hands_its_region_to_the_context() -> None:
    ctx = counter.instantiate(region="left").initialize()

    assert ctx.region == "left"
