# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Native viscosity models.

Importing this package registers the bundled native models with the
``viscosityModel`` plugin interface (registration is an import side-effect of
each model module).
"""

from .newtonian import newtonian

__all__ = ["newtonian"]
