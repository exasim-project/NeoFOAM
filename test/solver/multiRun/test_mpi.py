# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The layout of the regions on MPI ranks and the checks before a run, without MPI.

``test_regions_on_processes.py`` runs them under ``mpirun``.
"""

import sys
from pathlib import Path

import pytest

from neofoam.solver.multiRun.mpi import (
    MpiSetupError,
    RegionLayout,
    check_decomposed,
    check_size,
    import_mpi,
    launched_size,
    under_mpi,
)


def test_region_layout_puts_the_regions_on_blocks_of_ranks_in_stepping_order() -> None:
    layout = RegionLayout({"left": 2, "solo": 1, "right": 3})

    assert layout.total == 6
    assert [layout.region_of(rank) for rank in range(6)] == [
        "left",
        "left",
        "solo",
        "right",
        "right",
        "right",
    ]
    assert [layout.index(region) for region in ("left", "solo", "right")] == [0, 1, 2]
    assert str(layout) == "left: ranks 0-1, solo: rank 2, right: ranks 3-5"


def test_region_layout_runs_every_region_in_its_own_world() -> None:
    layout = RegionLayout({"left": 2, "right": 1})

    assert layout.argv("left") == ["multiRun", "-parallel", "-world", "left"]
    assert layout.argv("right") == ["multiRun", "-parallel", "-world", "right"]


def test_region_layout_has_no_region_beyond_its_ranks() -> None:
    with pytest.raises(MpiSetupError, match="rank 3 runs no region"):
        RegionLayout({"left": 2, "right": 1}).region_of(3)


def test_check_size_asks_for_the_processes_of_all_regions() -> None:
    check_size(RegionLayout({"left": 2, "right": 3}), 5)

    with pytest.raises(MpiSetupError) as error:
        check_size(RegionLayout({"left": 2, "right": 3}), 4)

    assert str(error.value) == (
        "the case was started on 4 MPI processes, but the regions of "
        "system/controlDict.yaml run on 2 (left) + 3 (right) = 5; start it with "
        "mpirun -np 5, or change the regions' processes"
    )


def test_check_decomposed_wants_a_processor_directory_per_process(tmp_path: Path) -> None:
    for n in range(2):
        (tmp_path / "left" / f"processor{n}").mkdir(parents=True)
    (tmp_path / "right" / "processor0").mkdir(parents=True)

    check_decomposed(RegionLayout({"left": 2}), tmp_path)
    with pytest.raises(MpiSetupError) as error:
        check_decomposed(RegionLayout({"left": 2, "right": 3}), tmp_path)

    assert "region right runs on 3 processes, but right/ lacks processor1, processor2" in str(
        error.value
    )
    assert "decomposePar -case right" in str(error.value)


def test_check_decomposed_refuses_more_processor_directories(tmp_path: Path) -> None:
    for n in range(3):
        (tmp_path / "left" / f"processor{n}").mkdir(parents=True)

    with pytest.raises(MpiSetupError, match=r"left/processor2 exists"):
        check_decomposed(RegionLayout({"left": 2}), tmp_path)


def test_check_decomposed_wants_processor0_for_a_region_on_one_process(tmp_path: Path) -> None:
    (tmp_path / "left").mkdir()

    with pytest.raises(MpiSetupError, match="region left runs on 1 process, but left/ lacks"):
        check_decomposed(RegionLayout({"left": 1}), tmp_path)

    (tmp_path / "left" / "processor0").mkdir()
    check_decomposed(RegionLayout({"left": 1}), tmp_path)


@pytest.mark.parametrize(
    ("variables", "size"),
    [
        ({}, 1),
        ({"OMPI_COMM_WORLD_SIZE": "1"}, 1),
        ({"OMPI_COMM_WORLD_SIZE": "4"}, 4),
        ({"PMI_SIZE": "3"}, 3),
    ],
)
def test_launched_size_reads_the_launcher(
    monkeypatch: pytest.MonkeyPatch, variables: dict, size: int
) -> None:
    for variable in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "MV2_COMM_WORLD_SIZE"):
        monkeypatch.delenv(variable, raising=False)
    for variable, value in variables.items():
        monkeypatch.setenv(variable, value)

    assert launched_size() == size
    assert under_mpi() == (size > 1)


def test_import_mpi_names_the_extra_without_mpi4py(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "mpi4py", None)
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "2")

    with pytest.raises(ImportError, match=r"2 MPI processes, but mpi4py .* neofoam\[mpi\]"):
        import_mpi()
