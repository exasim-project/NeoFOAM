# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""OpenFOAM-specific utilities and per-model dictionary base classes.

Currently exports:

- ``fvSchemes`` / ``fvSolution`` — ``BaseConfig`` base classes
  bound to ``system/fvSchemes`` and ``system/fvSolution``. Register
  via ``spec.config(fvSchemes)`` to receive a per-spec subclass that
  operations extend via ``@<Subclass>.add(...)``.
- ``schemes`` — typed scheme unions (DdtScheme, DivScheme, …) used as
  the value types when ``.add(...)`` injects fields into a subclass.
"""

from neofoam.foam.fv_configs import fvSchemes, fvSolution

__all__ = ["fvSchemes", "fvSolution"]
