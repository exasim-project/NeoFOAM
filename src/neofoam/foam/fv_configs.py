# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""
FvSchemes and FvSolution configs — loaded via @IOStrategy.

Uses ``extra="allow"`` so all sections from the OpenFOAM dictionary
are read automatically by the OpenFOAM strategy.
"""

from neofoam.io import BaseConfig, IOStrategy, OF


@IOStrategy(OF("system/fvSchemes"))
class FvSchemesConfig(BaseConfig):
    """Full fvSchemes dictionary loaded via IOStrategy.

    All sections (ddtSchemes, divSchemes, etc.) are read as extra fields.
    """

    model_config = {"extra": "allow"}


@IOStrategy(OF("system/fvSolution"))
class FvSolutionConfig(BaseConfig):
    """Full fvSolution dictionary loaded via IOStrategy.

    All sections (solvers, PIMPLE, etc.) are read as extra fields.
    """

    model_config = {"extra": "allow"}
