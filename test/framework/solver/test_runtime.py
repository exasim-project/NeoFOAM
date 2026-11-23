# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""``SolverRuntime.finalize`` calls ``finalize()`` on the models that have one."""

from typing import Any

from neofoam.framework.context import Context
from neofoam.framework.solver import Solver


class Releasing:
    """A model that logs its finalize."""

    def __init__(self, log: list[str]) -> None:
        self.log = log

    def finalize(self) -> None:
        self.log.append("finalize")


finalizing = Solver("finalizing")


@finalizing.initializer
def _initialize_finalizing(self: Any) -> Context:
    log: list[str] = []
    return Context(fields={}, models={"log": log, "releasing": Releasing(log)})


def test_runtime_finalize_skips_the_models_without_finalize() -> None:
    runtime = finalizing.instantiate()
    ctx = runtime.initialize()

    runtime.finalize(ctx)

    assert ctx.models["log"] == ["finalize"]
