# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The sweep.csv + params.yaml parameter space behind generated Snakefiles."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from neofoam.tooling.workflow.paramspace import (
    KeyedDim,
    YamlParamSpace,
    read_params_yaml,
    read_sweep_csv,
    write_if_changed,
    write_params_yaml,
    write_sweep_csv,
)


class _Transport(BaseModel):
    transportModel: str = "Newtonian"
    nu: float = 1e-5


_ROWS = [
    {"case": "long_nu1", "control": "long", "transport": "nu1"},
    {"case": "long_nu2", "control": "long", "transport": "nu2"},
    {"case": "short_nu1", "control": "short", "transport": "nu1"},
]

_VARIANTS = {
    "control": {"long": {"endTime": 5.0}, "short": {"endTime": 1.0}},
    "transport": {
        "nu1": {"transportModel": "Newtonian", "nu": 1e-5},
        "nu2": {"transportModel": "Newtonian", "nu": 2e-5},
    },
}


def _write_space(tmp_path: Path) -> tuple[Path, Path]:
    sweep, params = tmp_path / "sweep.csv", tmp_path / "params.yaml"
    write_sweep_csv(sweep, _ROWS)
    write_params_yaml(params, _VARIANTS)
    return sweep, params


def test_sweep_csv_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "sweep.csv"
    write_sweep_csv(path, _ROWS)
    assert path.read_text().splitlines()[0] == "case,control,transport"
    assert read_sweep_csv(path) == _ROWS


def test_sweep_csv_missing_case_column(tmp_path: Path) -> None:
    path = tmp_path / "sweep.csv"
    path.write_text("name,transport\na,nu1\n")
    with pytest.raises(ValueError, match="missing the 'case' column"):
        read_sweep_csv(path)


def test_sweep_csv_duplicate_case(tmp_path: Path) -> None:
    path = tmp_path / "sweep.csv"
    path.write_text("case,transport\na,nu1\na,nu2\n")
    with pytest.raises(ValueError, match="duplicate case name 'a'"):
        read_sweep_csv(path)


def test_params_yaml_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "params.yaml"
    write_params_yaml(path, _VARIANTS)
    assert read_params_yaml(path) == _VARIANTS


def test_params_yaml_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "params.yaml"
    path.write_text("- a\n- b\n")
    with pytest.raises(ValueError, match="expected a mapping"):
        read_params_yaml(path)
    path.write_text("transport:\n- nu1\n")
    with pytest.raises(ValueError, match="dimension 'transport'"):
        read_params_yaml(path)


def test_write_if_changed_preserves_mtime(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    assert write_if_changed(path, "{}\n") is True
    mtime = path.stat().st_mtime_ns
    assert write_if_changed(path, "{}\n") is False
    assert path.stat().st_mtime_ns == mtime
    assert write_if_changed(path, '{"a": 1}\n') is True


def test_paramspace_resolves_cases(tmp_path: Path) -> None:
    space = YamlParamSpace(*_write_space(tmp_path), models={"transport": _Transport})
    assert space.cases == ["long_nu1", "long_nu2", "short_nu1"]
    assert space.dims == ["control", "transport"]
    assert space.wildcard_pattern == "{case}"
    assert space.instance_patterns == space.cases
    assert space.config_for("long_nu2") == {
        "control": {"endTime": 5.0},
        "transport": {"transportModel": "Newtonian", "nu": 2e-5},
    }
    assert space.config_for("long_nu2", dims=["transport"]) == {
        "transport": {"transportModel": "Newtonian", "nu": 2e-5}
    }
    assert space.instance({"case": "short_nu1"})["control"] == {"endTime": 1.0}


def test_config_for_rejects_unknown_case_and_dimension(tmp_path: Path) -> None:
    space = YamlParamSpace(*_write_space(tmp_path))
    with pytest.raises(KeyError, match="unknown case 'nope'"):
        space.config_for("nope")
    with pytest.raises(ValueError, match="unknown dimension 'mesh'"):
        space.config_for("long_nu1", dims=["mesh"])


def test_instance_requires_a_case_wildcard(tmp_path: Path) -> None:
    space = YamlParamSpace(*_write_space(tmp_path))
    with pytest.raises(KeyError, match="carry no 'case'"):
        space.instance({})


def test_paramspace_validation_errors(tmp_path: Path) -> None:
    sweep, params = _write_space(tmp_path)

    write_params_yaml(params, {"transport": _VARIANTS["transport"]})
    with pytest.raises(ValueError, match="sweep column 'control'"):
        YamlParamSpace(sweep, params)

    write_params_yaml(
        params, {**_VARIANTS, "transport": {"nu1": _VARIANTS["transport"]["nu1"]}}
    )
    with pytest.raises(ValueError, match="unknown variant 'nu2'"):
        YamlParamSpace(sweep, params)

    bad = {**_VARIANTS, "transport": {"nu1": {"nu": "not-a-number"}}}
    write_sweep_csv(sweep, [_ROWS[0]])
    write_params_yaml(params, bad)
    with pytest.raises(ValueError, match="'transport.nu1' failed validation"):
        YamlParamSpace(sweep, params, models={"transport": _Transport})

    write_sweep_csv(sweep, [])
    write_params_yaml(params, _VARIANTS)
    with pytest.raises(ValueError, match="has no rows"):
        YamlParamSpace(sweep, params)


def test_materialize_writes_per_case_rule_configs(tmp_path: Path) -> None:
    space = YamlParamSpace(*_write_space(tmp_path))
    out = tmp_path / "configs"
    changed = space.materialize({"setup": space.dims}, out_dir=out)
    assert sorted(p.relative_to(out).as_posix() for p in changed) == [
        "long_nu1/setup.json",
        "long_nu2/setup.json",
        "short_nu1/setup.json",
    ]
    payload = json.loads((out / "long_nu1" / "setup.json").read_text())
    assert payload == space.config_for("long_nu1")

    # Unchanged content is not rewritten; a variant edit rewrites only affected cases.
    assert space.materialize({"setup": space.dims}, out_dir=out) == []
    space.variants["transport"]["nu2"]["nu"] = 3e-5
    changed = space.materialize({"setup": space.dims}, out_dir=out)
    assert [p.parent.name for p in changed] == ["long_nu2"]


def test_variant_of_routes_cases_to_variants(tmp_path: Path) -> None:
    space = YamlParamSpace(*_write_space(tmp_path))
    assert space.variant_of({"case": "long_nu2"}, "transport") == "nu2"
    assert space.variant_of({"case": "short_nu1"}, "control") == "short"

    class _Wildcards:
        """Snakemake-style wildcards: provides .get, not __getitem__."""

        def get(self, key: str, default: object = None) -> object:
            return {"case": "long_nu1"}.get(key, default)

    assert space.variant_of(_Wildcards(), "transport") == "nu1"

    with pytest.raises(KeyError, match="unknown case 'nope'"):
        space.variant_of({"case": "nope"}, "transport")
    with pytest.raises(KeyError, match="carry no 'case'"):
        space.variant_of({}, "transport")
    with pytest.raises(ValueError, match="unknown dimension 'mesh'"):
        space.variant_of({"case": "long_nu1"}, "mesh")


def test_materialize_dims_writes_per_variant_configs(tmp_path: Path) -> None:
    space = YamlParamSpace(*_write_space(tmp_path))
    out = tmp_path / "configs"
    changed = space.materialize_dims(["transport"], out_dir=out)
    assert sorted(p.relative_to(out).as_posix() for p in changed) == [
        "transport/nu1.json",
        "transport/nu2.json",
    ]
    assert json.loads((out / "transport" / "nu1.json").read_text()) == {
        "transportModel": "Newtonian",
        "nu": 1e-5,
    }
    # Unchanged content is not rewritten; one edited variant rewrites one file.
    assert space.materialize_dims(["transport"], out_dir=out) == []
    space.variants["transport"]["nu2"]["nu"] = 3e-5
    assert [p.name for p in space.materialize_dims(["transport"], out_dir=out)] == [
        "nu2.json"
    ]

    with pytest.raises(ValueError, match="unknown dimension 'mesh'"):
        space.materialize_dims(["mesh"], out_dir=out)


def test_keyed_present_dimension(tmp_path: Path) -> None:
    space = YamlParamSpace(*_write_space(tmp_path))
    axis = space.keyed("transport", out_dir=tmp_path / "configs")
    assert axis.names == ("nu1", "nu2")
    assert axis.space is space
    assert (tmp_path / "configs" / "transport" / "nu1.json").is_file()
    assert axis.of({"case": "long_nu2"}) == "nu2"


def test_keyed_absent_dimension_is_implicit_single_variant(tmp_path: Path) -> None:
    space = YamlParamSpace(*_write_space(tmp_path))
    axis = space.keyed("mesh", out_dir=tmp_path / "configs")
    assert axis == KeyedDim(dim="mesh", names=("base",), space=None)
    # The implicit variant materializes an empty payload so the keyed rule has
    # a concrete input file, and .of() routes every case to it.
    assert (tmp_path / "configs" / "mesh" / "base.json").read_text() == "{}\n"
    assert axis.of({"case": "long_nu1"}) == "base"
    assert axis.of({}) == "base"
