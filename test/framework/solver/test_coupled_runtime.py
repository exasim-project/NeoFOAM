# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

# NB: no ``from __future__ import annotations`` — the solver spec matches
# ``param.annotation is Context``.

"""A coupled region needs a solver that adds a coupling model.

``uncoupling`` adds no coupling model (``ctx.models["coupling"]``): run alone it
resolves its graph; as a region the case couples it raises, instead of running
the region uncoupled.
"""

from typing import Any

import pytest

from neofoam.framework.context import Context
from neofoam.framework.coupling import InProcessTransport, Link
from neofoam.framework.coupling.region import region_couplings
from neofoam.framework.operations import Operations, StepBuilder
from neofoam.framework.solver import Solver

uncoupling = Solver("uncoupling")


@uncoupling.initializer
def _initialize(self: Any) -> Context:
    return Context(fields={}, models={}, region=self.region)


@uncoupling.execution_graph_step
def _execution_graph(self: Any, ctx: Context) -> tuple[StepBuilder, Operations]:
    builder = StepBuilder()
    builder.step(self.operations["step"])
    return builder, Operations()


@uncoupling.operation()
def step(self: Any, ctx: Context) -> None:
    pass


def test_a_region_the_case_does_not_couple_resolves_without_a_coupling_model() -> None:
    runtime = uncoupling.instantiate(region="left")
    ctx = runtime.initialize()

    assert [op.operation_name for op in runtime.resolve_operations(ctx)] == ["step"]


def test_a_coupled_region_whose_solver_added_no_coupling_model_raises() -> None:
    link = Link("x", (("left", "p"), ("right", "p")))
    coupling = region_couplings(["left", "right"], [link], InProcessTransport())["left"]
    runtime = uncoupling.instantiate(region="left", coupling=coupling)
    ctx = runtime.initialize()

    with pytest.raises(
        RuntimeError,
        match=r"region left is coupled in regionCoupling.yaml, but solver uncoupling added no "
        r"coupling model; add ctx.models\['coupling'\] = Coupling\(self.coupling\) in initialize",
    ):
        runtime.resolve_operations(ctx)
