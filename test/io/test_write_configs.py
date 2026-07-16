# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for ``write_configs`` (``neofoam.io.write_configs``).

``write_configs`` groups configs by their target file and deep-merges co-owners
into one write, so multi-owner files (e.g. ``constant/transportProperties`` from
both ``TransportPropertiesConfig`` and ``BoussinesqConfig``) keep every
contribution while the writer still clears-and-rewrites each file (so re-saving a
single config drops keys it no longer carries). Writing goes through pybFoam.
"""

from __future__ import annotations

from pathlib import Path

import pybFoam as pyf

from neofoam.framework.solver.configurations import configurations
from neofoam.io import write_configs
from neofoam.io.base import BaseConfig
from neofoam.io.decorator import OF, IOStrategy
from neofoam.solver.incompressibleFluid.incompressibleFluid import (
    incompressibleFluid,
)


def _cfgs():
    return configurations(incompressibleFluid)


def test_write_configs_writes_field_and_dict_files(tmp_path: Path) -> None:
    cfgs = _cfgs()
    u = cfgs["UFieldConfig"].model_validate(
        {
            "internalField": "uniform (0 0 0)",
            "boundaryField": {"walls": {"type": "noSlip"}},
        }
    )
    cd = cfgs["ControlDictConfig"].model_validate({"endTime": 1.0, "deltaT": 0.1})

    report = write_configs([u, cd], case_dir=tmp_path)

    assert report["0/U"] == ["UFieldConfig"]
    assert report["system/controlDict"] == ["ControlDictConfig"]
    assert (tmp_path / "0" / "U").exists()
    assert (tmp_path / "system" / "controlDict").exists()
    text = (tmp_path / "0" / "U").read_text()
    assert "noSlip" in text and "boundaryField" in text


def test_write_configs_skips_instances_without_io_config(tmp_path: Path) -> None:
    from pydantic import BaseModel

    class NoIO(BaseModel):
        x: int = 1

    assert write_configs([NoIO()], case_dir=tmp_path) == {}


def test_write_configs_co_owners_accumulate(tmp_path: Path) -> None:
    """Two configs targeting one file accumulate — neither clobbers the other."""

    @IOStrategy(OF("constant/shared"))
    class _A(BaseConfig):
        alpha: int = 1

    @IOStrategy(OF("constant/shared"))
    class _B(BaseConfig):
        beta: int = 2

    write_configs([_A(alpha=11), _B(beta=22)], case_dir=tmp_path)

    root = pyf.dictionary.read(str(tmp_path / "constant" / "shared"))
    assert root.found("alpha") and root.found("beta")
    assert str(root.get[str]("alpha")) == "11"
    assert str(root.get[str]("beta")) == "22"


def test_write_configs_merges_multi_owner_file(tmp_path: Path) -> None:
    """Multi-owner ``constant/transportProperties`` keeps both contributors.

    Merging Boussinesq with Transport must not drop Transport's ``nu`` (a naive
    per-config clear-write would).
    """
    cfgs = _cfgs()
    transport = cfgs["TransportPropertiesConfig"].model_validate(
        {"transportModel": "Newtonian", "nu": 1e-5}
    )
    boussinesq = cfgs["BoussinesqConfig"].model_construct()
    assert (
        transport.io_config.file
        == boussinesq.io_config.file
        == "constant/transportProperties"
    )

    write_configs([transport, boussinesq], case_dir=tmp_path)

    # Transport's value survived Boussinesq's later write to the same file.
    reloaded = cfgs["TransportPropertiesConfig"].load(case_dir=tmp_path)
    assert reloaded.nu == 1e-5
    assert reloaded.transportModel == "Newtonian"


def test_dict_config_emits_foamfile_header(tmp_path: Path) -> None:
    cfgs = _cfgs()
    cd = cfgs["ControlDictConfig"].model_validate({"endTime": 1.0, "deltaT": 0.1})
    cd.save(case_dir=tmp_path)  # single-instance path also injects a header

    root = pyf.dictionary.read(str(tmp_path / "system" / "controlDict"))
    header = root.subDict("FoamFile")
    assert str(header.get[str]("class")) == "dictionary"
    assert str(header.get[str]("object")) == "controlDict"


def test_field_config_keeps_native_header(tmp_path: Path) -> None:
    cfgs = _cfgs()
    u = cfgs["UFieldConfig"].model_validate(
        {"boundaryField": {"walls": {"type": "noSlip"}}}
    )
    u.save(case_dir=tmp_path)

    root = pyf.dictionary.read(str(tmp_path / "0" / "U"))
    header = root.subDict("FoamFile")
    assert str(header.get[str]("class")) == "volVectorField"
    assert str(header.get[str]("object")) == "U"


def test_round_tripped_foamfile_header_is_written_first(tmp_path: Path) -> None:
    # A payload that came back through ``load(...)`` (AI-fill push, sweep seeds)
    # carries ``FoamFile`` as its LAST key; OpenFOAM rejects a file whose header
    # is not the leading entry, so the writer must hoist it to the top.
    cfgs = _cfgs()
    cd = cfgs["ControlDictConfig"].model_validate(
        {
            "endTime": 1.0,
            "deltaT": 0.1,
            "FoamFile": {
                "version": "2",
                "format": "ascii",
                "class": "dictionary",
                "object": "controlDict",
            },
        }
    )
    write_configs([cd], case_dir=tmp_path)

    text = (tmp_path / "system" / "controlDict").read_text()
    assert text.index("FoamFile") < text.index("endTime")
    # And it still parses as a valid OpenFOAM dict with one header.
    root = pyf.dictionary.read(str(tmp_path / "system" / "controlDict"))
    assert str(root.subDict("FoamFile").get[str]("object")) == "controlDict"
    assert text.count("FoamFile") == 1


def test_write_configs_writes_yaml_strategy_config(tmp_path: Path) -> None:
    # ``PreprocessConfig`` is a YAMLStrategy config (``system/preprocess.yaml``).
    # Before the YAML merged-write path, ``write_configs`` raised
    # ``NotImplementedError: no merged-write path for YAMLStrategy`` — so a case
    # authored through the MCP ``save_case`` tool could not emit its preprocess
    # enable-list. It must now write and round-trip like the OpenFOAM configs.
    from neofoam.framework.tools.graph import PreprocessConfig

    cfg = PreprocessConfig.model_validate(
        {
            "tools": [
                {"tool": "blockMesh"},
                {
                    "tool": "checkMesh",
                    "depends_on": ["blockMesh"],
                    "fail_on_error": False,
                },
            ]
        }
    )

    report = write_configs([cfg], case_dir=tmp_path)
    assert report["system/preprocess.yaml"] == ["PreprocessConfig"]

    path = tmp_path / "system" / "preprocess.yaml"
    assert path.is_file()
    # No OpenFOAM ``FoamFile`` header leaks into a YAML file.
    assert "FoamFile" not in path.read_text()

    loaded = PreprocessConfig.load(case_dir=tmp_path)
    assert [t["tool"] for t in loaded.tools] == ["blockMesh", "checkMesh"]
