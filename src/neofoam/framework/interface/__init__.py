# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""
neofoam.framework.interface — InterfaceSpec / Interface package.

Public API:
    InterfaceSpec   — immutable gather-point definition, generic in T
    Interface       — factory: Interface("name") -> InterfaceSpec
    BoundInterface  — spec bound to a live Context; calling it returns the fold
"""

from .spec import BoundInterface, Interface, InterfaceSpec

__all__ = [
    "BoundInterface",
    "Interface",
    "InterfaceSpec",
]
