# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Unit tests for the ``patch`` and ``line`` sources — the spec, not the sampling.

What a source is *declared* to be needs no mesh: the discriminated union resolves
it from a case file's ``type``, the sugar builds the same object, ``set_type``
decides which of the line spec's fields are required, and the spec is translated
into the ``pybFoam.sampling`` config with OpenFOAM's own spelling. Those are the
parts that break silently, so they are pinned here. Sampling a live mesh is
``test_sources_e2e.py``.

Reading a volume field off either field library moved to
:meth:`~neofoam.postprocess.node.InternalDataSet.from_field` and is pinned in
``test/postprocess/test_dataset.py``; the live NeoN read is
``test/solver/incompressibleFluidNeoN/test_post_process.py``.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from pydantic import ValidationError

from neofoam.postprocess.node import Source
from neofoam.postprocess.sources.fields import (
    OUT_OF_MESH,
    LineField,
    PatchField,
    _in_mesh,
    line,
    patch,
)


def _resolved(spec: dict[str, Any]) -> Source:
    """The source a case file's mapping selects from the ``Source`` family."""
    return cast(Source, cast(Any, Source).create(source=spec).source)


# --- the declarative front door ------------------------------------------


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        pytest.param(
            {"type": "patch", "field": "p", "patch": "movingWall"},
            PatchField(field="p", patch="movingWall"),
            id="patch",
        ),
        pytest.param(
            {
                "type": "line",
                "field": "U",
                "start": [0.0, 0.0, 0.005],
                "end": [0.0, 0.1, 0.005],
                "n_points": 5,
            },
            LineField(field="U", start=(0.0, 0.0, 0.005), end=(0.0, 0.1, 0.005), n_points=5),
            id="line_uniform",
        ),
        pytest.param(
            {
                "type": "line",
                "field": "p",
                "set_type": "cloud",
                "points": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            },
            LineField(field="p", set_type="cloud", points=[(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]),
            id="line_cloud",
        ),
    ],
)
def test_a_declared_source_resolves_to_its_class(spec: dict[str, Any], expected: Source) -> None:
    assert _resolved(spec) == expected


def test_the_patch_sugar_builds_the_declared_source() -> None:
    pipeline = patch("p", "movingWall")

    assert pipeline.source == _resolved({"type": "patch", "field": "p", "patch": "movingWall"})


def test_the_line_sugar_builds_a_uniform_set_between_the_two_points() -> None:
    pipeline = line("U", (0.0, 0.0, 0.005), (0.0, 0.1, 0.005), 5)

    assert pipeline.source == LineField(
        field="U", set_type="uniform", start=(0.0, 0.0, 0.005), end=(0.0, 0.1, 0.005), n_points=5
    )


# --- the line spec: which data each set type needs, and how it is spelled --


@pytest.mark.parametrize(
    ("set_type", "missing"),
    [
        pytest.param("uniform", "n_points", id="uniform_without_n_points"),
        pytest.param("cloud", "points", id="cloud_without_points"),
        pytest.param("polyLine", "points", id="polyline_without_points"),
        pytest.param("circle", "d_theta", id="circle_without_d_theta"),
    ],
)
def test_a_line_spec_missing_its_set_types_data_is_rejected(set_type: str, missing: str) -> None:
    complete: dict[str, Any] = {
        "field": "U",
        "set_type": set_type,
        "start": (0.0, 0.0, 0.0),
        "end": (0.0, 0.1, 0.0),
        "n_points": 5,
        "points": [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)],
        "origin": (0.0, 0.0, 0.0),
        "circle_axis": (0.0, 0.0, 1.0),
        "start_point": (0.1, 0.0, 0.0),
        "d_theta": 10.0,
    }
    del complete[missing]

    with pytest.raises(ValidationError, match=missing):
        LineField(**complete)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        pytest.param(
            LineField(field="U", start=(0.0, 0.0, 0.0), end=(0.0, 0.1, 0.0), n_points=5),
            {
                "type": "uniform",
                "axis": "distance",
                "start": [0.0, 0.0, 0.0],
                "end": [0.0, 0.1, 0.0],
                "nPoints": 5,
            },
            id="uniform",
        ),
        pytest.param(
            LineField(field="p", set_type="cloud", points=[(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]),
            {
                "type": "cloud",
                "axis": "distance",
                "points": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            },
            id="cloud",
        ),
        pytest.param(
            LineField(
                field="p",
                set_type="polyLine",
                points=[(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)],
                n_points=7,
            ),
            {
                "type": "polyLine",
                "axis": "distance",
                "points": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                "nPoints": 7,
            },
            id="polyline",
        ),
        pytest.param(
            LineField(
                field="p",
                set_type="circle",
                origin=(0.0, 0.0, 0.0),
                circle_axis=(0.0, 0.0, 1.0),
                start_point=(0.1, 0.0, 0.0),
                d_theta=45.0,
            ),
            {
                "type": "circle",
                "axis": "distance",
                "origin": [0.0, 0.0, 0.0],
                "circleAxis": [0.0, 0.0, 1.0],
                "startPoint": [0.1, 0.0, 0.0],
                "dTheta": 45.0,
            },
            id="circle",
        ),
    ],
)
def test_a_line_spec_becomes_the_sampling_config_openfoam_spells(
    source: LineField, expected: dict[str, Any]
) -> None:
    config = source._set_config()

    assert config.model_dump(exclude_none=True) == expected


# --- the out-of-mesh mask -------------------------------------------------


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        pytest.param(np.array([1.0, OUT_OF_MESH, -3.0]), [1, 0, 1], id="scalar"),
        pytest.param(
            np.array([[1.0, 0.0, 0.0], [OUT_OF_MESH, OUT_OF_MESH, OUT_OF_MESH]]),
            [1, 0],
            id="vector",
        ),
    ],
)
def test_a_point_the_sampler_could_not_evaluate_is_masked_out(
    values: "np.ndarray[Any, Any]", expected: list[int]
) -> None:
    # a mask is 0/1 labels, the way the kernels read one
    assert_array_equal(_in_mesh(values), np.array(expected))
