# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The solver's config schema covers every file a runnable laminar case needs.

Hermetic -- ``configurations`` is case-free and schema-only (no OpenFOAM). Guards
that the two mesh-input dicts are registered (so the wizard / MCP / agent fill
them) and that the whole laminar file set is owned by some config class.
"""

from neofoam.framework.solver.configurations import configurations
from neofoam.solver.incompressibleFluid import incompressibleFluid

# Every file the laminar incompressibleFluid skeleton must write to run
# preprocess (mesh) + solve.
_REQUIRED_FILES = {
    "system/controlDict",
    "system/blockMeshDict",
    "system/snappyHexMeshDict",
    "system/preprocess.yaml",
    "system/fvSchemes",
    "system/fvSolution",
    "constant/transportProperties",
    "constant/turbulenceProperties",
    "0/U",
    "0/p",
}


def test_mesh_dicts_are_registered() -> None:
    """The two mesh-input configs are surfaced by ``configurations``."""
    names = configurations(incompressibleFluid).names
    assert "BlockMeshDictConfig" in names
    assert "SnappyHexMeshDictConfig" in names


def test_mesh_dicts_bind_to_the_right_files() -> None:
    configs = configurations(incompressibleFluid)
    files = {cls.__name__: cls.io_config.file for cls in configs if cls.io_config}
    assert files["BlockMeshDictConfig"] == "system/blockMeshDict"
    assert files["SnappyHexMeshDictConfig"] == "system/snappyHexMeshDict"


def test_every_required_file_has_a_config() -> None:
    """No gap: each required case file is owned by at least one config class."""
    covered = {
        cls.io_config.file
        for cls in configurations(incompressibleFluid)
        if cls.io_config
    }
    missing = _REQUIRED_FILES - covered
    assert not missing, f"no config owns: {sorted(missing)}"
