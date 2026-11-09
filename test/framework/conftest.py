# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

"""Shared test helpers and BaseConfig stubs for framework tests."""

from typing import Any

import pytest

from neofoam.framework.coupling import InProcessTransport
from neofoam.io import BaseConfig


@pytest.fixture(autouse=True)
def _pickled_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every in-process transport pickles its messages: a message that is not plain data fails."""
    monkeypatch.setattr(InProcessTransport, "PICKLE", True)


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
