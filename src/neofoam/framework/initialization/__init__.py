# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

"""
3-Stage Initialization Framework (Hybrid Explicit Approach)

This module provides utilities for explicit 3-stage initialization:
- LOAD: Load configuration and data from files
- RESOLVE: Validate and connect models (inter-model dependencies)
- BUILD: Create lazy initializers for runtime structures

The framework uses explicit method calls instead of decorators:

Usage Example (Solver Initializer):
    @dataclass
    class MySolverInitializer:
        def load(self) -> dict[str, Any]:
            \"\"\"Load configuration from files.\"\"\"
            algorithm = Algorithm.from_file("system/fvSolution")
            return {"algorithm": algorithm}

        def resolve(self, config: ConfigContext) -> None:
            \"\"\"Configure inter-model dependencies.\"\"\"
            for model in self.optional_models:
                if hasattr(model, "resolve"):
                    model.resolve(config)

        def build(self) -> list[InitStep]:
            \"\"\"Create lazy initializers for runtime objects.\"\"\"
            return [
                field("U", create=lambda ctx: create_vector_field(ctx["mesh"])),
                model("transport", depends_on=["fields.U"], create=...),
            ]

Usage Example (Model):
    class MyModel(BaseModel):
        def load(self) -> dict[str, Any]:
            \"\"\"Load model configuration.\"\"\"
            return {}

        def resolve(self, config: ConfigContext) -> None:
            \"\"\"Configure dependencies with other models.\"\"\"
            pass

        def build(self) -> list[InitStep]:
            \"\"\"Create field initializers.\"\"\"
            return [field("T", create=...)]

Execution:
    from neofoam.framework.initialization.execution import execute_initialization

    # In solver.initialize()
    initializer = MySolverInitializer(argv)
    config_items = initializer.load()

    config = ConfigContext()
    for key, value in config_items.items():
        config.register(key, value)

    initializer.resolve(config)
    lazy_inits = initializer.build()
    ctx = execute_initialization(lazy_inits)
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
