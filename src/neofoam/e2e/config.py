# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Completeness gate for an assembled case.

:func:`require_configs` fails -- naming the offending file -- if any file a laminar
``incompressibleFluid`` run needs is absent or does not load. The case itself is
authored the config-driven way (fill the solver's ``BaseConfig`` models and
``write_configs`` them; see ``test/e2e/cases/fill_tube_bank.py``); this module only
verifies the result.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["MissingConfigError", "REQUIRED_FILES", "require_configs"]

# Every file the laminar incompressibleFluid skeleton needs to run preprocess + solve.
REQUIRED_FILES: tuple[str, ...] = (
    "system/controlDict",
    "system/fvSchemes",
    "system/fvSolution",
    "system/blockMeshDict",
    "system/snappyHexMeshDict",
    "system/preprocess.yaml",
    "constant/transportProperties",
    "constant/turbulenceProperties",
    "0/U",
    "0/p",
)


class MissingConfigError(Exception):
    """A required preprocess+solve file is absent or fails to load."""

    def __init__(self, relpath: str, reason: str) -> None:
        self.relpath = relpath
        self.reason = reason
        super().__init__(f"required config {relpath!r} {reason}")


def _loaders() -> dict[str, Any]:
    """Lazily resolve the config loaders (pybFoam-backed; gated path only)."""
    from neofoam.solver.incompressibleFluid.configs import ControlDictConfig
    from neofoam.turbulence.config import TurbulencePropertiesConfig
    from neofoam.viscosity.config import TransportPropertiesConfig

    return {
        "system/controlDict": ControlDictConfig,
        "constant/transportProperties": TransportPropertiesConfig,
        "constant/turbulenceProperties": TurbulencePropertiesConfig,
    }


def require_configs(case_dir: str | Path, *, check_loadable: bool = True) -> None:
    """Raise :class:`MissingConfigError` for the first missing/unloadable required file.

    The presence check is pure filesystem (hermetic). When ``check_loadable`` (the
    default), the config-bound files are additionally loaded via their solver loaders --
    which require pybFoam, so they are imported lazily here.
    """
    case = Path(case_dir)
    for rel in REQUIRED_FILES:
        if not (case / rel).is_file():
            raise MissingConfigError(rel, "is absent")
    if not check_loadable:
        return
    for rel, loader in _loaders().items():
        try:
            loader.load(case_dir=case)
        except Exception as exc:  # file present but does not validate
            raise MissingConfigError(rel, f"failed to load: {exc}") from exc
