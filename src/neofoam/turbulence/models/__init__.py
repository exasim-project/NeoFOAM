# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Native turbulence models.

Importing this package registers the bundled native models with the
``momentumTransportModel`` plugin interface (registration is an import
side-effect of each model module).
"""

from .laminar import laminar

__all__ = ["laminar"]
