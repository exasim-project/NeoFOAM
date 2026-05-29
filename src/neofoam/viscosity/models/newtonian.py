# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Trivial native viscosity model: ``Newtonian`` (constant viscosity).

Skeleton model that exercises the full plugin-registration and selection path
without any transport physics. ``correct`` is a no-op (viscosity is constant).
Real non-Newtonian models (CrossPowerLaw, BirdCarreau, …) follow this shape,
adding ``@build`` / ``@operation`` physics in later work.
"""

from pathlib import Path
from typing import Any, Optional

from ..interface import Model, viscosityModel

__all__ = ["newtonian", "NewtonianModel"]

newtonian = Model("Newtonian").register_with(viscosityModel)


@newtonian.detect
def detect() -> bool:
    """``Newtonian`` is always available as the constant-viscosity model."""
    return True


@newtonian.load
def load(_case_dir: Path, _instance_id: Optional[str]) -> None:
    """Skeleton: Newtonian carries no config yet."""
    return None


class NewtonianModel:
    """Constant-viscosity model satisfying the ``ViscosityModel`` Protocol."""

    def nu(self) -> Any:
        """Kinematic viscosity — placeholder in the skeleton."""
        return 0.0

    def correct(self) -> None:
        """Newtonian viscosity is constant — nothing to recompute."""
        return None
