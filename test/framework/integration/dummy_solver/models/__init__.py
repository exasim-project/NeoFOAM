# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Models for DummySolver.
"""

from .dummy_model import DummyModel
from .model1 import ThermalModel

__all__ = ["DummyModel", "ThermalModel"]
