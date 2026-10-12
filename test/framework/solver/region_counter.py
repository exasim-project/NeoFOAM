# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""A test-only solver that reads one increment per region of ``cases/regions``."""

from pathlib import Path
from typing import Any

from neofoam.framework.context import Context
from neofoam.framework.solver import Solver
from neofoam.io import YAML, BaseConfig, IOStrategy

CASE = Path(__file__).parent / "cases" / "regions"


@IOStrategy(YAML("constant/increment.yaml"))
class IncrementConfig(BaseConfig):
    increment: float


counter = Solver("counter")


@counter.initializer
def _initialize(self: Any) -> Context:
    config = IncrementConfig.load(case_dir=CASE, region=self.region)
    return Context(fields={"total": 0.0}, models={"increment": config.increment})
