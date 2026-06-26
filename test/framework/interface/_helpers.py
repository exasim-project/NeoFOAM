# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Shared spec factories for interface tests."""

from __future__ import annotations

import builtins
from typing import cast

from neofoam.framework.interface import Interface, InterfaceSpec

VGREAT = 1.0e30


def _make_min_spec(name: str) -> InterfaceSpec[float]:
    """Return a fresh float min-fold spec with @combine registered."""
    spec: InterfaceSpec[float] = Interface(name)

    @spec.combine
    def _fold(values: object) -> float:
        return cast(float, builtins.min(values, default=VGREAT))  # type: ignore[call-overload]

    return spec


def _make_sum_spec(name: str) -> InterfaceSpec[int]:
    """Return a fresh int sum-fold spec with @combine registered."""
    spec: InterfaceSpec[int] = Interface(name)

    @spec.combine
    def _fold(values: object) -> int:
        return cast(int, sum(values))  # type: ignore[call-overload]

    return spec
