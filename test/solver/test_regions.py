# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""``neofoam.solver.regions`` on the files of ``cases/coupledEntries``, ``cases/participants``.

``written`` is ``T`` as OpenFOAM's ``fixedValue`` writes it: no ``coupled`` entry.
"""

import gzip
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from neofoam.solver.regions import (
    ControlDictConfig,
    CoupledEntriesWriteHook,
    coupled_entries,
    keep_coupled_entries,
    restore_coupled_entries,
)

CASES = Path(__file__).parent / "cases" / "coupledEntries"
PARTICIPANTS = Path(__file__).parent / "cases" / "participants"
PATCH_TYPES = {"coupledTemperature": "calculated", "coupledHeatFlux": "calculated"}
WRITTEN_TYPES = {"coupledTemperature": "fixedValue"}
ENTRY = b"        coupled         coupledTemperature;\n"


def test_coupled_entries_maps_each_coupled_patch_to_its_condition() -> None:
    assert coupled_entries(CASES / "coupled", PATCH_TYPES) == {"interface": "coupledTemperature"}


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("unknownCondition", "patch interface is coupled as 'coupledPressure'; known: "),
        ("wrongType", "has type fixedValue; a coupledTemperature patch is 'calculated'"),
    ],
)
def test_coupled_entries_rejects_an_unknown_condition_or_type(case: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        coupled_entries(CASES / case, PATCH_TYPES)


def test_participants_take_a_plain_name_as_one_process() -> None:
    config = ControlDictConfig.load(case_dir=PARTICIPANTS / "entries")

    assert config.solvers() == {"left": "a", "right": "b"}
    assert config.processes() == {"left": 1, "right": 3}


def test_participants_default_a_region_to_one_process() -> None:
    config = ControlDictConfig.load(case_dir=PARTICIPANTS / "default_processes")

    assert config.processes() == {"left": 1}


@pytest.mark.parametrize("name", ["zero_processes", "unknown_entry", "no_solver"])
def test_participants_reject_an_invalid_entry(name: str) -> None:
    with pytest.raises(ValidationError):
        ControlDictConfig.load(case_dir=PARTICIPANTS / name)


def test_restore_coupled_entries_adds_the_entry_to_that_patch_only(tmp_path: Path) -> None:
    path = shutil.copy(CASES / "written", tmp_path / "T")

    restore_coupled_entries(path, {"interface": "coupledTemperature"})

    assert coupled_entries(path, WRITTEN_TYPES) == {"interface": "coupledTemperature"}
    assert path.read_bytes().replace(ENTRY, b"") == (CASES / "written").read_bytes()


def test_restore_coupled_entries_adds_an_entry_once(tmp_path: Path) -> None:
    path = shutil.copy(CASES / "written", tmp_path / "T")

    restore_coupled_entries(path, {"interface": "coupledTemperature"})
    restore_coupled_entries(path, {"interface": "coupledTemperature"})

    assert path.read_bytes().count(b"coupled ") == 1


def test_restore_coupled_entries_edits_a_compressed_file(tmp_path: Path) -> None:
    (tmp_path / "T.gz").write_bytes(gzip.compress((CASES / "written").read_bytes()))

    restore_coupled_entries(tmp_path / "T", {"interface": "coupledTemperature"})

    assert not (tmp_path / "T").exists()
    assert b"    {\n" + ENTRY + b"        type            fixedValue;\n" in gzip.decompress(
        (tmp_path / "T.gz").read_bytes()
    )


def test_restore_coupled_entries_rejects_a_patch_the_file_lacks(tmp_path: Path) -> None:
    path = shutil.copy(CASES / "written", tmp_path / "T")

    with pytest.raises(ValueError, match="no patch wall in boundaryField"):
        restore_coupled_entries(path, {"wall": "coupledTemperature"})


class _OpenFoamWrite:
    """Writes ``written`` as T into ``<case>/<time>``, as OpenFOAM does."""

    def __init__(self, case: Path) -> None:
        self.case = case
        self.time = "0.5"

    def timeName(self) -> str:
        return self.time

    def write_fields(self, fields: Any) -> None:
        (self.case / self.time).mkdir(parents=True)
        shutil.copy(CASES / "written", self.case / self.time / "T")


def test_keep_coupled_entries_leaves_an_uncoupled_writer_alone(tmp_path: Path) -> None:
    inner = _OpenFoamWrite(tmp_path)
    writer = SimpleNamespace(hook=inner)

    keep_coupled_entries(writer, inner, tmp_path, {"T": {}})

    assert writer.hook is inner


def test_keep_coupled_entries_adds_them_after_each_write(tmp_path: Path) -> None:
    inner = _OpenFoamWrite(tmp_path)
    writer = SimpleNamespace(hook=inner)

    keep_coupled_entries(writer, inner, tmp_path, {"T": {"interface": "coupledTemperature"}})

    assert isinstance(writer.hook, CoupledEntriesWriteHook) and writer.hook.inner is inner
    for time in ("0.5", "1"):
        inner.time = time
        writer.hook.write_fields({})
        assert coupled_entries(tmp_path / time / "T", WRITTEN_TYPES) == {
            "interface": "coupledTemperature"
        }, f"written T at {time}"
