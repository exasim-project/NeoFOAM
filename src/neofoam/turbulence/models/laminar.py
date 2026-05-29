# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Trivial native turbulence model: ``laminar`` (no turbulence).

Skeleton model that exercises the full plugin-registration and selection path
without any turbulence physics. ``nut`` is zero and ``correct`` is a no-op. Real
models (kEpsilon, kOmegaSST, …) follow this shape, adding ``@build`` /
``@operation`` physics in later work.
"""

from pathlib import Path
from typing import Any, Optional

from ..interface import Model, turbulenceModel

__all__ = ["laminar", "LaminarModel"]

laminar = Model("laminar").register_with(turbulenceModel)


@laminar.detect
def detect() -> bool:
    """``laminar`` is always available as the no-op model."""
    return True


@laminar.load
def load(_case_dir: Path, _instance_id: Optional[str]) -> None:
    """Skeleton: laminar carries no config yet."""
    return None


class LaminarModel:
    """No-op turbulence model satisfying the ``TurbulenceModel`` Protocol."""

    def nut(self) -> Any:
        """Turbulent viscosity — zero for laminar flow."""
        return 0.0

    def nu(self) -> Any:
        """Laminar viscosity — placeholder in the skeleton."""
        return 0.0

    def divDevReff(self, U: Any) -> Any:
        """Not implemented in the skeleton (no momentum coupling yet)."""
        raise NotImplementedError("laminar divDevReff is not implemented yet")

    def correct(self) -> None:
        """No turbulence fields to advance for laminar flow."""
        return None
