# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Interface model family of incompressibleVoFNeoN: the interface normal and curvature."""

from .grad_alpha import gradAlphaInterface
from .interfaceModel import InterfaceModel, interfaceModelNeoN

__all__ = ["InterfaceModel", "gradAlphaInterface", "interfaceModelNeoN"]
