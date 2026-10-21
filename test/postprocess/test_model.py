# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Spec for the postProcess *core* Model (backend-agnostic).

The model is driven the way the framework drives it — LOAD from a case
directory, BUILD the InitStep, then call the operation with a real
:class:`~neofoam.framework.context.Context` carrying a real
:class:`~neofoam.algorithms.solution_loop.loop_state.LoopState` (the write
policies read exactly that ``StepView``). The ``time`` column is the writer's
shortest spelling of the step time (``0``, ``0.1``), not its full repr. Only the
simulation backend is faked:
a source reads ``internalField()`` off a field and ``C()``/``V()`` off the mesh,
nothing else, so no OpenFOAM is needed. The cadence case declares one table per
cadence, which is what makes "a table is evaluated on its own write steps"
observable in the CSV files.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from neofoam.algorithms.solution_loop.loop_state import LoopState
from neofoam.framework.context import Context
from neofoam.postprocess.model import PostProcessor, build, post_process, postProcess
from neofoam.postprocess.table import TableSet

CASES = Path(__file__).parent / "cases"

CELL_VOLUMES = np.array([0.5, 0.25, 1.0])
PRESSURE = np.array([1.0, 2.0, 3.0])


class FakeField:
    """A volume field: a source reads only its internal values."""

    def __init__(self, values: np.ndarray) -> None:
        self._values = values

    def internalField(self) -> np.ndarray:
        return self._values


class FakeMesh:
    """An fvMesh: the cell source reads only cell centres and cell volumes."""

    def C(self) -> FakeField:
        return FakeField(np.zeros((len(CELL_VOLUMES), 3)))

    def V(self) -> np.ndarray:
        return CELL_VOLUMES


def _processor(case_dir: Path) -> PostProcessor:
    """The runtime object the framework builds for a case's tables."""
    steps = build(postProcess._load_func(case_dir, "main"))
    return cast(PostProcessor, steps[0].initializer({}))


def _ctx(processor: PostProcessor, index: int, *, start_time: float = 0.0) -> Context:
    return Context(
        models={"post_processor": processor},
        fields={"p": FakeField(PRESSURE)},
        mesh=FakeMesh(),
        time=LoopState(
            value=0.1 * index,
            delta_t=0.1,
            end_time=1.0,
            index=index,
            start_time=start_time,
        ),
    )


def _rows(path: Path) -> list[list[str]]:
    with path.open(newline="") as handle:
        return list(csv.reader(handle))


# --- the Model ------------------------------------------------------------


def test_modelspec_is_a_full_core_model() -> None:
    assert postProcess.name == "postProcess"
    assert postProcess._load_func is not None
    op_names = {meta["name"] for _, meta in postProcess._operations}
    assert op_names == {"post_process"}


def test_build_emits_the_post_processor(tmp_path: Path) -> None:
    steps = build(TableSet(case_dir=tmp_path))
    assert [step.name for step in steps] == ["models.post_processor"]
    assert isinstance(steps[0].initializer({}), PostProcessor)


# --- running the operation over a few steps -------------------------------


def test_each_table_is_written_on_its_own_write_steps(tmp_path: Path) -> None:
    case_dir = tmp_path / "cadence"
    shutil.copytree(CASES / "cadence", case_dir)
    processor = _processor(case_dir)

    for index in range(3):
        post_process(None, _ctx(processor, index))

    assert _rows(case_dir / "postProcessing/every_step.csv") == [
        ["time", "every_step"],
        ["0", "4.0"],
        ["0.1", "4.0"],
        ["0.2", "4.0"],
    ]
    assert [row[0] for row in _rows(case_dir / "postProcessing/every_second_step.csv")] == [
        "time",
        "0",
        "0.2",
    ]


def test_two_tables_sharing_the_default_run_time_cadence_both_write(tmp_path: Path) -> None:
    case_dir = tmp_path / "shared_cadence"
    shutil.copytree(CASES / "shared_cadence", case_dir)
    processor = _processor(case_dir)

    for index in range(3):
        post_process(None, _ctx(processor, index))

    assert [row[0] for row in _rows(case_dir / "postProcessing/first.csv")] == [
        "time",
        "0.1",
        "0.2",
    ]
    assert [row[0] for row in _rows(case_dir / "postProcessing/second.csv")] == [
        "time",
        "0.1",
        "0.2",
    ]


def test_a_pipeline_that_does_not_end_in_an_aggregator_names_the_table_and_the_node(
    tmp_path: Path,
) -> None:
    case_dir = tmp_path / "no_aggregator"
    shutil.copytree(CASES / "no_aggregator", case_dir)
    processor = _processor(case_dir)

    with pytest.raises(TypeError, match=r"'not_aggregated'.*node 'scale'"):
        post_process(None, _ctx(processor, 0))


def test_case_without_tables_writes_nothing(tmp_path: Path) -> None:
    processor = _processor(tmp_path)

    post_process(None, _ctx(processor, 0))

    assert not (tmp_path / "postProcessing").exists()


def test_a_restart_appends_to_the_existing_csv(tmp_path: Path) -> None:
    case_dir = tmp_path / "cadence"
    shutil.copytree(CASES / "cadence", case_dir)
    post_process(None, _ctx(_processor(case_dir), 0))

    post_process(None, _ctx(_processor(case_dir), 1, start_time=0.1))

    assert [row[0] for row in _rows(case_dir / "postProcessing/every_step.csv")] == [
        "time",
        "0",
        "0.1",
    ]
