# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""What multiRun checks before it runs under MPI, without MPI.

``regionSolvers`` gives each region its ``processes``; :class:`RegionLayout`
puts the regions on the ranks of MPI_COMM_WORLD in blocks, in stepping order,
as OpenFOAM's ``-world`` numbers them. The checks of
:mod:`neofoam.solver.multiRun.mpi` refuse a case that cannot run on the
processes it was started on, each with a message that says what to change.
``test_multiRun_mpi.py`` runs them under ``mpirun``.
"""

import importlib
import sys
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from neofoam.framework.solver import Solver
from neofoam.solver.heatConductionNeoN import heatConductionNeoN
from neofoam.solver.multiRun.mpi import (
    MpiSetupError,
    RegionLayout,
    check_decomposed,
    check_size,
    check_sub_communicator,
    check_uncoupled,
    import_mpi,
    launched_size,
    under_mpi,
)
from neofoam.solver.neon_runtime import neon_takes_a_communicator
from neofoam.solver.regions import ControlDictConfig

CASES = Path(__file__).parent / "cases"
MULTI_RUN = importlib.import_module("neofoam.solver.multiRun.multiRun")


def _config(tmp_path: Path, regions: dict) -> ControlDictConfig:
    (tmp_path / "system").mkdir(exist_ok=True)
    (tmp_path / "system" / "controlDict.yaml").write_text(
        yaml.safe_dump({"endTime": 1.0, "deltaT": 1.0, "regionSolvers": regions})
    )
    return ControlDictConfig.load(case_dir=tmp_path)


def test_region_solvers_take_a_plain_name_as_one_process(tmp_path: Path) -> None:
    config = _config(tmp_path, {"left": "a", "right": {"solver": "b", "processes": 3}})

    assert config.solvers() == {"left": "a", "right": "b"}
    assert config.processes() == {"left": 1, "right": 3}


def test_region_solvers_default_a_region_to_one_process(tmp_path: Path) -> None:
    config = _config(tmp_path, {"left": {"solver": "a"}})

    assert config.processes() == {"left": 1}


@pytest.mark.parametrize(
    "entry",
    [
        {"solver": "a", "processes": 0},
        {"solver": "a", "processes": 2, "threads": 1},
        {"processes": 2},
    ],
)
def test_region_solvers_reject_an_invalid_entry(tmp_path: Path, entry: dict) -> None:
    with pytest.raises(ValidationError):
        _config(tmp_path, {"left": entry})


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


def test_check_sub_communicator_refuses_a_solver_that_needs_every_process() -> None:
    bound = Solver("bound")
    bound.sub_communicator_limitation = "it talks to every process"
    specs = {"left": Solver("free"), "right": bound}

    check_sub_communicator(RegionLayout({"left": 1}), {"left": specs["left"]})
    with pytest.raises(MpiSetupError) as error:
        check_sub_communicator(RegionLayout({"left": 1, "right": 1}), specs)

    assert str(error.value) == (
        "region right runs on 1 of the 2 MPI processes, but its solver bound cannot run "
        "on part of them: it talks to every process"
    )


def test_heat_conduction_neon_runs_on_part_of_the_processes_if_neon_can() -> None:
    limitation = heatConductionNeoN.sub_communicator_limitation

    if neon_takes_a_communicator():
        assert limitation is None
    else:
        assert limitation is not None
        assert "this NeoN build communicates on MPI_COMM_WORLD" in limitation


def test_check_uncoupled_refuses_a_coupled_case(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(CASES / "coupled_slabs")

    with pytest.raises(MpiSetupError, match="the MPI comm channel is not implemented yet") as error:
        check_uncoupled(["left", "right"])

    assert "interface (left - right)" in str(error.value)


def test_check_uncoupled_passes_an_uncoupled_case(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(CASES / "two_regions")

    check_uncoupled(["left", "right"])


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


def test_run_takes_the_mpi_path_under_mpirun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    taken: list[str] = []
    _config(tmp_path, {"left": "x"})
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(MULTI_RUN, "run_mpi", lambda config: taken.append("mpi") or {})
    monkeypatch.setattr(MULTI_RUN, "run_threads", lambda config: taken.append("threads") or {})
    monkeypatch.delenv("PMI_SIZE", raising=False)
    monkeypatch.delenv("MV2_COMM_WORLD_SIZE", raising=False)

    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "1")
    MULTI_RUN.run()
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "2")
    MULTI_RUN.run()

    assert taken == ["threads", "mpi"]
