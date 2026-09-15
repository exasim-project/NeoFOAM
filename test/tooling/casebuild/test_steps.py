# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Unit tests for the built-in steps: ``patch``, ``configs`` (meshing → ``test_meshing``).

Each step is exercised against a materialized :class:`CaseDir` staged from the
committed ``cavity`` template. Configs are built the way production does, from
``configurations(incompressibleFluid)``.

The ``regexKeys`` template ships an ``fvSolution`` holding double-quoted regex keys.
OpenFOAM parses such a key into an *unquoted* pattern keyword, so patching or
removing one lands on the existing entry only if the lookup uses that stored form.
Only sub-dict values are covered: pybFoam's scalar ``set`` strips a keyword's quotes,
so a regex-keyed *leaf* cannot be written at all.
"""

import shutil
from pathlib import Path

import pybFoam as pyf
import pytest

from neofoam.framework.solver.configurations import configurations
from neofoam.solver.incompressibleFluid.incompressibleFluid import incompressibleFluid
from neofoam.tooling.casebuild import CaseDir, configs, patch

CAVITY = Path(__file__).parent / "cases" / "cavity"
REGEX_KEYS = Path(__file__).parent / "cases" / "regexKeys"


def _staged(tmp_path: Path, template: Path = CAVITY) -> CaseDir:
    dst = tmp_path / "case"
    shutil.copytree(template, dst)
    return CaseDir(dst)


def _solvers(case: CaseDir) -> pyf.dictionary:
    return pyf.dictionary.read(str(case.path / "system" / "fvSolution")).subDict("solvers")


def _keys(d: pyf.dictionary) -> list[str]:
    return [str(k) for k in d.toc()]


def test_patch_merges_dict_and_kwargs_with_kwargs_winning(tmp_path: Path) -> None:
    case = _staged(tmp_path)
    patch("system/controlDict", {"endTime": 1.0, "deltaT": 0.5}, endTime=2.0)(case)
    d = pyf.dictionary.read(str(case.path / "system" / "controlDict"))
    assert d.get_scalar("endTime") == pytest.approx(2.0)  # kwargs override the dict
    assert d.get_scalar("deltaT") == pytest.approx(0.5)


def test_patch_missing_file_raises(tmp_path: Path) -> None:
    case = _staged(tmp_path)
    with pytest.raises(FileNotFoundError):
        patch("system/doesNotExist", endTime=1.0)(case)


def test_patch_remove_drops_the_key(tmp_path: Path) -> None:
    case = _staged(tmp_path)
    assert pyf.dictionary.read(str(case.path / "system" / "controlDict")).found("deltaT")
    patch("system/controlDict", remove=["deltaT"])(case)
    d = pyf.dictionary.read(str(case.path / "system" / "controlDict"))
    assert not d.found("deltaT")
    assert d.found("endTime")  # neighbours untouched


def test_patch_sets_and_removes_in_one_step(tmp_path: Path) -> None:
    # patch subsumes the old unset: one call forks a base both ways at once.
    case = _staged(tmp_path)
    patch("system/controlDict", {"endTime": 2.0}, remove=["deltaT"])(case)
    d = pyf.dictionary.read(str(case.path / "system" / "controlDict"))
    assert d.get_scalar("endTime") == pytest.approx(2.0)
    assert not d.found("deltaT")


def test_patch_creates_quoted_regex_key_via_dotted_address(tmp_path: Path) -> None:
    case = _staged(tmp_path)  # cavity: solvers {} is empty
    patch(
        "system/fvSolution",
        **{'solvers.".*Final"': {"solver": "PBiCGStab", "relTol": 0}},
    )(case)
    solvers = _solvers(case)
    assert _keys(solvers) == [".*Final"]  # one key, stored as the parsed pattern
    assert solvers.subDict(".*Final").get_word("solver") == "PBiCGStab"


def test_patch_creates_quoted_regex_key_via_nested_mapping(tmp_path: Path) -> None:
    case = _staged(tmp_path)
    patch("system/fvSolution", solvers={'".*Final"': {"solver": "PBiCGStab", "relTol": 0}})(case)
    assert _keys(_solvers(case)) == [".*Final"]


def test_patch_writes_regex_key_with_its_quotes_verbatim(tmp_path: Path) -> None:
    # Without the quotes the written file is not even re-readable, so assert on text.
    case = _staged(tmp_path)
    patch("system/fvSolution", **{'solvers.".*Final"': {"solver": "PBiCGStab"}})(case)
    assert '".*Final"' in (case.path / "system" / "fvSolution").read_text()


def test_patch_unquoted_dotted_key_descends_into_subdicts(tmp_path: Path) -> None:
    case = _staged(tmp_path, REGEX_KEYS)
    patch("system/fvSolution", **{"solvers.p.relTol": 0.05})(case)
    p = _solvers(case).subDict("p")
    assert p.get_scalar("relTol") == pytest.approx(0.05)
    assert p.get_word("solver") == "GAMG"  # siblings untouched


def test_patch_updates_existing_regex_key_without_duplicating_it(tmp_path: Path) -> None:
    case = _staged(tmp_path, REGEX_KEYS)
    patch("system/fvSolution", **{'solvers.".*Final"': {"solver": "PCG", "relTol": 0}})(case)
    solvers = _solvers(case)
    assert _keys(solvers) == ["p", ".*Final", "(U|k|epsilon)"]  # no second ".*Final"
    assert solvers.subDict(".*Final").get_word("solver") == "PCG"


def test_patch_removes_quoted_regex_key(tmp_path: Path) -> None:
    case = _staged(tmp_path, REGEX_KEYS)
    patch("system/fvSolution", remove=['solvers."(U|k|epsilon)"'])(case)
    assert _keys(_solvers(case)) == ["p", ".*Final"]


def test_configs_writes_config_file(tmp_path: Path) -> None:
    case = _staged(tmp_path)
    cfgs = configurations(incompressibleFluid)
    transport = cfgs["TransportPropertiesConfig"].model_validate(
        {"transportModel": "Newtonian", "nu": 0.01}
    )
    configs(transport)(case)
    assert (case.path / "constant" / "transportProperties").is_file()
