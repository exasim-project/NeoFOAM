# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""Alpha-advection scheme family for incompressibleVoF.

Importing this package registers every scheme with the ``advectionModel`` family
(the ``register_with`` calls run as import side-effects), so
``advectionModel.detect_and_create()`` can select the active one.
"""

from .advectionModel import advectionModel
from .models.iso_advector import iso_advector
from .models.mules import mules

__all__ = ["advectionModel", "mules", "iso_advector"]
