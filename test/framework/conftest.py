# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

"""Shared test helpers and BaseConfig stubs for framework tests."""

from typing import Any

from neofoam.io import BaseConfig


class MaxIterations:
    """Callable that returns True for a fixed number of calls, then False."""

    def __init__(self, max_iters: int = 5) -> None:
        self.max_iters = max_iters
        self.current_iter = 0

    def __call__(self, ctx: Any) -> bool:
        self.current_iter += 1
        if self.current_iter <= self.max_iters:
            return True
        return False


# ---------------------------------------------------------------------------
# Shared BaseConfig stubs
# ---------------------------------------------------------------------------


class MyConfig(BaseConfig):
    x: int = 1


class OtherConfig(BaseConfig):
    y: int = 2


class StepConfig(BaseConfig):
    factor: float = 0.01


class StubConfig(BaseConfig):
    factor: float = 2.0
