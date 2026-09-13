# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Unit tests for the CsvWriter — the default writer and the file-IO of the package.

Everything runs in ``tmp_path``: the writer owns the file layout (it appends
``.csv`` to the stem it is opened on, ``time`` first column, header once,
directory created lazily so a case with no tables leaves nothing behind) and a
restart continues an existing file instead of truncating it.

Which rank writes is decided by ``pybFoam.Pstream.master()``; the rank is faked
by patching that binding so the branch runs without a decomposed case
(``test_reduce.py`` fakes ``parRun`` the same way).
"""

from __future__ import annotations

from pathlib import Path

import pybFoam as pyf
import pytest

from neofoam.postprocess.node import AggregatedDataSet
from neofoam.postprocess.writers.csv import CsvWriter


def _result(*values: float) -> AggregatedDataSet:
    headers = ["volume_p"] if len(values) == 1 else [f"volume_U_{i}" for i in range(len(values))]
    return AggregatedDataSet(name="volume_p", headers=headers, rows=[list(values)])


def _writer(path_stem: Path, *, append: bool = False) -> CsvWriter:
    """A writer opened the way the PostProcessor opens one: on a suffix-less stem."""
    writer = CsvWriter()
    writer.open(path_stem, append=append)
    return writer


def test_header_is_written_once_and_every_write_appends_one_row(tmp_path: Path) -> None:
    writer = _writer(tmp_path / "volume_p")

    writer.write(0.1, _result(2.5))
    writer.write(0.2, _result(3.0))

    assert (tmp_path / "volume_p.csv").read_text().splitlines() == [
        "time,volume_p",
        "0.1,2.5",
        "0.2,3.0",
    ]


def test_the_time_column_is_written_as_the_shortest_spelling_of_the_time(tmp_path: Path) -> None:
    # what ten loop steps of 0.005 accumulate to as a double
    writer = _writer(tmp_path / "volume_p")

    writer.write(0.049999999999999996, _result(2.5))

    assert (tmp_path / "volume_p.csv").read_text().splitlines()[1] == "0.05,2.5"


def test_a_multi_column_result_keeps_time_as_the_first_column(tmp_path: Path) -> None:
    writer = _writer(tmp_path / "volume_U")

    writer.write(0.1, _result(1.0, 2.0, 3.0))

    assert (tmp_path / "volume_U.csv").read_text().splitlines() == [
        "time,volume_U_0,volume_U_1,volume_U_2",
        "0.1,1.0,2.0,3.0",
    ]


def test_a_table_name_carrying_a_dot_keeps_it_in_the_file_name(tmp_path: Path) -> None:
    # ``with_suffix`` would turn alpha.water into alpha.csv
    _writer(tmp_path / "alpha.water").write(0.1, _result(2.5))

    assert (tmp_path / "alpha.water.csv").exists()


def test_the_directory_is_created_only_on_the_first_write(tmp_path: Path) -> None:
    path = tmp_path / "postProcessing" / "volume_p.csv"
    writer = _writer(path.parent / "volume_p")

    assert not path.parent.exists()

    writer.write(0.1, _result(2.5))

    assert path.exists()


def test_a_second_writer_truncates_the_file_by_default(tmp_path: Path) -> None:
    _writer(tmp_path / "volume_p").write(0.1, _result(2.5))

    _writer(tmp_path / "volume_p").write(0.2, _result(3.0))

    assert (tmp_path / "volume_p.csv").read_text().splitlines() == ["time,volume_p", "0.2,3.0"]


def test_append_continues_an_existing_file_without_a_second_header(tmp_path: Path) -> None:
    _writer(tmp_path / "volume_p").write(0.1, _result(2.5))

    _writer(tmp_path / "volume_p", append=True).write(0.2, _result(3.0))

    assert (tmp_path / "volume_p.csv").read_text().splitlines() == [
        "time,volume_p",
        "0.1,2.5",
        "0.2,3.0",
    ]


def test_append_writes_the_header_when_the_file_is_missing(tmp_path: Path) -> None:
    _writer(tmp_path / "volume_p", append=True).write(0.1, _result(2.5))

    assert (tmp_path / "volume_p.csv").read_text().splitlines() == ["time,volume_p", "0.1,2.5"]


@pytest.mark.parametrize(
    ("master", "written"),
    [(True, True), (False, False)],
    ids=["master_rank_writes", "other_rank_writes_nothing"],
)
def test_only_the_master_rank_touches_the_filesystem(
    master: bool, written: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(pyf.Pstream, "master", lambda: master)
    path = tmp_path / "postProcessing" / "volume_p.csv"

    _writer(path.parent / "volume_p").write(0.1, _result(2.5))

    assert path.exists() is written
    assert path.parent.exists() is written
