# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The Boussinesq model's own configs: ``constant/g`` and its fvSchemes/fvSolution keys.

The buoyant momentum/pressure variants live in the PIMPLE algorithm file, but the
entries only they read (``rhok`` / ``p_rgh``) belong to the Boussinesq slices, so a
non-buoyant case is neither asked for them nor written with them.
"""

from pathlib import Path

import pytest

pytest.importorskip("pybFoam")  # the OpenFOAM write/read path needs pybFoam

import pybFoam as pyf

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
    write_configs([GravityConfig()], case_dir=tmp_path)
    root = pyf.dictionary.read(str(tmp_path / "constant" / "g"))
    assert str(root.get[str]("dimensions")) == "[ 0 1 -2 0 0 0 0 ]"
    assert str(root.get[str]("value")) == "( 0 -9.81 0 )"


def _default_keys(*config_names: str) -> dict[str, set[str]]:
    """The entry names per section in the merged form defaults of *config_names*."""
    cfg = configurations(incompressibleFluid)
    keys: dict[str, set[str]] = {}
    for name in config_names:
        for section, entries in (cfg[name].form_defaults() or {}).items():
            keys.setdefault(section, set()).update(entries)
    return keys


@pytest.mark.parametrize("config_name", ["Pimple_fvSchemes", "Pimple_fvSolution"])
def test_pimple_defaults_carry_no_buoyancy_keys(config_name: str) -> None:
    names = {key for entries in _default_keys(config_name).values() for key in entries}
    assert [key for key in names if "rhok" in key or "p_rgh" in key] == []


def test_boussinesq_schemes_complete_the_buoyant_key_set() -> None:
    assert _default_keys("Pimple_fvSchemes", "boussinesq_fvSchemes") == {
        "ddtSchemes": {"ddt(U)", "default"},
        "divSchemes": {"div(phi,U)", "div((nuEff*dev2(T(grad(U)))))", "div(phi,T)"},
        "gradSchemes": {"grad(U)", "grad(p)", "grad(rhok)", "grad(p_rgh)", "grad(T)"},
        "laplacianSchemes": {
            "laplacian(nuEff,U)",
            "laplacian(rAU,p)",
            "laplacian(rAUf,p_rgh)",
            "default",
        },
        "interpolationSchemes": {
            "flux(HbyA)",
            "interpolate(rAU)",
            "dotInterpolate(S,U_0)",
            "flux(U)",
        },
        "snGradSchemes": {"snGrad(p)", "snGrad(rhok)", "snGrad(p_rgh)"},
    }


def test_boussinesq_solution_completes_the_buoyant_solver_set() -> None:
    solvers = _default_keys("Pimple_fvSolution", "boussinesq_fvSolution")["solvers"]
    assert solvers == {"U", "UFinal", "p", "pFinal", "p_rgh", "p_rghFinal", "T", "TFinal"}
