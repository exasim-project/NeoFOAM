# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""2-stage initialization of the multiRun solver.

LOAD  → the YAML controlDict and the framework ``solutionLoop`` fed with it
BUILD → loop state + engine; the regions are initialized by the solver afterwards
"""

from pathlib import Path
from typing import Any, Optional

from neofoam.algorithms.solution_loop.solution_loop import solutionLoop
from neofoam.framework.initialization import (
    InitializerBuilder,
    InitStep,
    LoadResult,
    StagedInitRunner,
    StagedInitSpec,
)
from neofoam.framework.initialization import model as init_model
from neofoam.framework.model import ModelRuntime

from .configs import ControlDictConfig


def create_init(case_dir: Optional[Path] = None) -> StagedInitRunner:
    """Build a fresh :class:`StagedInitRunner` for one multiRun case."""
    spec_builder = StagedInitSpec.build("multiRun")
    resolved_case_dir = case_dir or Path(".")

    @spec_builder.load
    def load_config() -> LoadResult:
        config = ControlDictConfig.load(case_dir=resolved_case_dir)
        loop = ModelRuntime(spec=solutionLoop, name="solutionLoop_main", config=config)
        return LoadResult(core_models=[loop, config], optional_models=[])

    @spec_builder.build
    def build_lazy(core_models: list[Any], optional_models: list[Any]) -> list[InitStep]:
        loop, _config = core_models
        builder = InitializerBuilder()
        builder.add_core_models([("solution_loop_model", loop)])
        builder.add(
            init_model("solutionLoop", lambda _ctx: loop, depends_on=["models.solution_loop"])
        )
        return builder.build()

    return StagedInitRunner(spec_builder.finalize())
