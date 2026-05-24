# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Shared physics/numerics model primitives reused across solvers."""

from .stability_criteria import CFLCondition

__all__ = ["CFLCondition"]
