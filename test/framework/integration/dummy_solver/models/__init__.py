# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Models for DummySolver.
"""

from .dummy_model import DummyModelInterface, Model
from .model1 import model1
from .model2 import model2
from .model3 import model3
from .model4 import model4

__all__ = ["DummyModelInterface", "Model", "model1", "model2", "model3", "model4"]
