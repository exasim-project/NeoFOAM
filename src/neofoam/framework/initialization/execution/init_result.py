# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Result type produced by executing a single :class:`InitStep`."""

from dataclasses import dataclass
from typing import Any

from ..init_step import InitCategory


@dataclass(frozen=True)
class InitResult:
    """Outcome of one :class:`InitStep` execution.

    Carries the step's ``name`` and ``category`` alongside the produced
    ``value`` so downstream routing can dispatch on category rather than
    parsing the name prefix.
    """

    name: str
    category: InitCategory
    value: Any
    write: bool = False  # field flagged for persistence (auto-write)
