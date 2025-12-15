# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

"""
Initialization Stage Enumeration

Defines the three stages of the initialization framework.
"""

from enum import Enum


class InitializationStage(str, Enum):
    """
    Enumeration of the three initialization stages.

    LOAD: Load configuration and data from files
    RESOLVE_DEPENDENCIES: Validate and connect models (inter-model dependencies)
    BUILD: Initialize runtime structures (fields, matrices, etc.)
    """

    LOAD = "LOAD"
    RESOLVE_DEPENDENCIES = "RESOLVE_DEPENDENCIES"
    BUILD = "BUILD"
