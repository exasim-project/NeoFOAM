# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Solver Factory - State management for solver initialization.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SolverState:
    """
    Shared state container for solver initialization.

    This dataclass stores state shared across the 3-stage initialization:
    - core_models: Required models for the solver
    - optional_models: Optional physics/turbulence models
    - configs: Configuration dictionary
    """

    core_models: list[Any] = field(default_factory=list)
    optional_models: list[Any] = field(default_factory=list)
    configs: dict[str, Any] = field(default_factory=dict)
