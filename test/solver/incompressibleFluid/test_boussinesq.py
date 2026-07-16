# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""GravityConfig — the ``constant/g`` the Boussinesq buoyancy build reads at init."""

from pathlib import Path

import pytest

pytest.importorskip("pybFoam")  # the OpenFOAM write/read path needs pybFoam

from neofoam.framework.solver.configurations import configurations  # noqa: E402
from neofoam.io import write_configs  # noqa: E402
from neofoam.solver.incompressibleFluid.incompressibleFluid import (  # noqa: E402
    incompressibleFluid,
)
from neofoam.solver.incompressibleFluid.models.boussinesq import (  # noqa: E402
    GravityConfig,
)


def test_gravity_config_is_a_registered_config_bound_to_constant_g() -> None:
    """The buoyancy model owns ``constant/g`` so it surfaces in the config schema."""
    cfg = configurations(incompressibleFluid)
    assert "GravityConfig" in cfg.names
    assert cfg["GravityConfig"].io_config is not None
    assert cfg["GravityConfig"].io_config.file == "constant/g"


def test_gravity_config_writes_uniform_dimensioned_vector_field(tmp_path: Path) -> None:
    """Defaults write Earth gravity in −y with the field class OpenFOAM expects."""
    write_configs([GravityConfig()], case_dir=tmp_path)
    text = (tmp_path / "constant" / "g").read_text()
    assert "class           uniformDimensionedVectorField;" in text
    # dimensions / value are unquoted OpenFOAM tokens, not quoted strings
    assert "dimensions      [ 0 1 -2 0 0 0 0 ];" in text
    assert "value           ( 0 -9.81 0 );" in text


def test_gravity_config_round_trips_tokens_to_python_lists(tmp_path: Path) -> None:
    """``dimensions``/``value`` read back from the bracket/paren tokens as lists."""
    write_configs([GravityConfig(value=[0.0, 0.0, -9.81])], case_dir=tmp_path)
    loaded = GravityConfig.load(case_dir=tmp_path)
    assert loaded.dimensions == [0, 1, -2, 0, 0, 0, 0]
    assert loaded.value == [0.0, 0.0, -9.81]


def test_gravity_config_openfoam_parses_written_file(tmp_path: Path) -> None:
    """pybFoam parses the written tokens (dimensionSet + vector), not raw strings."""
    import pybFoam as pyf

    write_configs([GravityConfig()], case_dir=tmp_path)
    root = pyf.dictionary.read(str(tmp_path / "constant" / "g"))
    assert str(root.get[str]("dimensions")) == "[ 0 1 -2 0 0 0 0 ]"
    assert str(root.get[str]("value")) == "( 0 -9.81 0 )"
