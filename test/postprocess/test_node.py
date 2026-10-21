# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Unit tests for the post-processing data contract: the datasets, Source/Node, Pipeline.

The value layer is deliberately pybFoam-free: a source only ever calls
``internalField()`` on a registered field and ``C()``/``V()`` on the mesh, so the
fakes below stand in for the whole backend and a real
:class:`~neofoam.framework.context.Context` carries them (the genuine object,
built the way a solver builds it, rather than a duck-typed stand-in). The
datasets themselves — the ``with_*`` helpers and
:meth:`~neofoam.postprocess.node.InternalDataSet.from_field` — are pinned in
``test_dataset.py``.
"""

from __future__ import annotations

from typing import Any, Iterator, Literal

import numpy as np
import pytest

from neofoam.core.plugin_system import PluginSystem
from neofoam.framework.context import Context
from neofoam.postprocess.node import (
    AggregatedData,
    AggregatedDataSet,
    InternalDataSet,
    Node,
    Pipeline,
    Source,
)
from neofoam.postprocess.nodes.aggregators import Sum, VolIntegrate
from neofoam.postprocess.nodes.binning import Directional
from neofoam.postprocess.nodes.debug import Print
from neofoam.postprocess.nodes.field_functions import Mag
from neofoam.postprocess.nodes.selectors import Not, Sphere
from neofoam.postprocess.sources.fields import InternalField, field

CELL_CENTRES = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
CELL_VOLUMES = np.array([0.5, 0.25, 1.0])
PRESSURE = np.array([1.0, 2.0, 3.0])


class FakeField:
    """A volume field: the source reads only its internal values."""

    def __init__(self, values: np.ndarray) -> None:
        self._values = values

    def internalField(self) -> np.ndarray:
        return self._values


class FakeMesh:
    """An fvMesh: the cell source reads only cell centres and cell volumes."""

    def C(self) -> FakeField:
        return FakeField(CELL_CENTRES)

    def V(self) -> np.ndarray:
        return CELL_VOLUMES


class RecordingNode(Node):
    """Appends its label to a shared log, then passes the dataset on."""

    label: str
    # ``Any`` so pydantic hands the caller's list through; a ``list[str]`` field
    # would be validated into a copy and the test could not read the log back.
    log: Any

    def compute(self, dataset: Any) -> Any:
        self.log.append(self.label)
        return dataset


class SelfAggregating(Source):
    """A source that is already a table (the solver residuals), so nothing may follow it."""

    type: Literal["self_aggregating"] = "self_aggregating"

    def resolve(self, ctx: Context) -> AggregatedDataSet:
        return AggregatedDataSet(name="residuals", values=[AggregatedData(value=1.0)])


def _ctx() -> Context:
    return Context(fields={"p": FakeField(PRESSURE)}, models={}, mesh=FakeMesh())


@pytest.fixture
def doubling_node() -> Iterator[type[Node]]:
    """A node class registered *after* import — the extension path a case script
    takes. Unregistered again so the shared ``Node`` union stays as it was."""

    class Doubling(Node):
        type: Literal["doubling"] = "doubling"
        factor: float = 2.0

        def compute(self, dataset: Any) -> Any:
            return dataset.with_field(np.asarray(dataset.field) * self.factor)

    Node.register(Doubling)
    yield Doubling
    PluginSystem.remove_plugin_model("Node", Doubling)


# --- composing a pipeline -------------------------------------------------


def test_field_starts_a_pipeline_on_the_internal_source() -> None:
    pipeline = field("p")

    assert isinstance(pipeline.source, InternalField)
    assert pipeline.source.field == "p"
    assert pipeline.steps == []


def test_or_appends_the_nodes_in_order() -> None:
    pipeline = field("p") | RecordingNode(label="first", log=[]) | VolIntegrate()

    assert [type(step) for step in pipeline.steps] == [RecordingNode, VolIntegrate]


def test_or_returns_a_new_pipeline_and_leaves_the_original_untouched() -> None:
    base = field("p")

    extended = base | VolIntegrate()

    assert extended is not base
    assert base.steps == []
    assert len(extended.steps) == 1


# --- evaluating a pipeline ------------------------------------------------


def test_compute_runs_the_source_then_every_step_in_order() -> None:
    log: list[str] = []
    first = RecordingNode(label="first", log=log)
    second = RecordingNode(label="second", log=log)

    result = (field("p") | first | second).compute(_ctx())

    assert log == ["first", "second"]
    assert isinstance(result, InternalDataSet)
    np.testing.assert_allclose(result.field, PRESSURE, rtol=1e-12)


def test_a_source_that_aggregates_by_itself_refuses_a_step_after_it() -> None:
    aggregating = Pipeline(source=SelfAggregating(), steps=[VolIntegrate()])

    with pytest.raises(TypeError, match=r"'self_aggregating'.*'volIntegrate'"):
        aggregating.compute(_ctx())


def test_internal_field_resolves_values_and_cell_geometry() -> None:
    dataset = InternalField(field="p").resolve(_ctx())

    assert dataset.name == "p"
    np.testing.assert_allclose(dataset.field, PRESSURE, rtol=1e-12)
    np.testing.assert_allclose(dataset.geometry.positions(), CELL_CENTRES, rtol=1e-12)
    np.testing.assert_allclose(dataset.geometry.volumes(), CELL_VOLUMES, rtol=1e-12)


def test_internal_field_of_an_unregistered_field_names_the_available_fields() -> None:
    with pytest.raises(KeyError, match=r"'U' is not registered; available: \['p'\]"):
        InternalField(field="U").resolve(_ctx())


def test_a_node_after_an_aggregator_names_both_of_them() -> None:
    aggregated = field("p") | Sum(name="p_sum") | Mag()

    with pytest.raises(TypeError, match=r"'sum'.*'mag'"):
        aggregated.compute(_ctx())


def test_a_print_node_may_follow_an_aggregator() -> None:
    result = (field("p") | Sum(name="p_sum") | Print()).compute(_ctx())

    assert isinstance(result, AggregatedDataSet)


# --- the plugin families --------------------------------------------------


def test_internal_field_parses_from_a_dict_by_its_type_string() -> None:
    selected: Any = Source.create(  # type: ignore[attr-defined]
        source={"type": "internal", "field": "p"}
    )

    assert isinstance(selected.source, InternalField)
    assert selected.source.field == "p"


def test_a_node_registered_after_import_parses_from_a_dict(doubling_node: type[Node]) -> None:
    selected: Any = Node.create(  # type: ignore[attr-defined]
        node={"type": "doubling", "factor": 3.0}
    )

    assert isinstance(selected.node, doubling_node)
    assert selected.node.factor == 3.0


def test_a_dumped_pipeline_validates_back_into_the_same_pipeline() -> None:
    pipeline = (
        field("U")
        | Not(region=Sphere(center=(0.0, 0.0, 0.0), radius=0.1))
        | Directional(bins=[0.5], direction=(1.0, 0.0, 0.0))
        | Sum(name="U_sum")
    )

    assert Pipeline.model_validate(pipeline.model_dump()) == pipeline


def test_a_pipeline_keeps_the_concrete_node_classes() -> None:
    pipeline = Pipeline(source=InternalField(field="p"), steps=[VolIntegrate(name="volume_p")])

    step = pipeline.steps[0]
    assert isinstance(step, VolIntegrate)
    assert step.name == "volume_p"
