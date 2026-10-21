# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Spec for the write-back: the declared values reach the case's 0/ fields.

**Integration, real OpenFOAM.** The whole point of the module is the pybFoam
seam, so there is nothing to fake: ``cases/box`` is a unit box cut into 4x4x1
cells whose ``0/`` holds deliberately *wrong* uniform values (``alpha.water``
0.5, ``U`` ``(9 9 9)``), so every assertion below fails unless setFields wrote
something. Cell centres sit at x = 0.125/0.375/0.625/0.875, so the case's
``system/setFields.yaml`` region ``x <= 0.5`` is the two left-hand columns —
eight of the sixteen cells, the pre-written expectation in ``_IN_REGION``.

The zero-copy check is a test rather than a comment because the whole design
rests on it: if ``np.asarray(field.internalField())`` ever stopped aliasing the
OpenFOAM memory, every assignment here would silently write to a copy and the
fields on disk would be unchanged.

Values are exact — an assignment is a copy, not an arithmetic result — except
the round trip through the ASCII file, which is compared with the case's
``writePrecision``-independent literals at ``assert_allclose`` defaults.

Every test copies the case into ``tmp_path`` and runs from inside it: the tool
reads and writes the time directory relative to the working directory, like
every preprocessing tool's dict file. A few tests load their declaration from a
*different* case directory (``cases/overlapping``, ``cases/region_only``) and
apply it to the same box mesh: the declaration is a file, the mesh is the
fixture's, and keeping the variants apart keeps ``cases/box`` the one case the
end-to-end tests run.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import numpy as np
import pybFoam as pyf
import pytest
from numpy.testing import assert_allclose
from pybFoam.meshing import generate_blockmesh

from neofoam.postprocess import Box
from neofoam.preprocess import apply_set_fields
from neofoam.preprocess.script import set_fields_for_case

CASES = Path(__file__).parent / "cases"

#: ``cases/box``: the eight cells the declared region selects, x fastest.
_IN_REGION = np.array([True, True, False, False] * 4)


@pytest.fixture
def box(tmp_path: Path) -> Iterator[Any]:
    """``cases/box`` copied into tmp_path, meshed, with the cwd inside it."""
    case_dir = tmp_path / "box"
    shutil.copytree(CASES / "box", case_dir)
    previous = Path.cwd()
    os.chdir(case_dir)
    try:
        time = pyf.Time(pyf.argList(["setFields", "-case", str(case_dir)]))
        yield generate_blockmesh(time, pyf.dictionary.read("system/blockMeshDict"))
    finally:
        os.chdir(previous)


def test_the_internal_field_is_a_writable_zero_copy_view(box: Any) -> None:
    field = pyf.volScalarField.read_field(box, "alpha.water", False)
    view = np.asarray(field.internalField())

    view[0] = 3.25

    assert view.flags.writeable
    assert np.asarray(field.internalField())[0] == 3.25


def test_a_declared_region_gets_its_value_and_the_rest_the_default(box: Any) -> None:
    setup = set_fields_for_case(Path("."))

    apply_set_fields(box, setup.defaults, setup.regions)

    alpha = np.asarray(pyf.volScalarField.read_field(box, "alpha.water", False).internalField())
    assert_allclose(alpha, np.where(_IN_REGION, 1.0, 0.0), err_msg="cases/box: alpha.water")


def test_a_vector_field_takes_the_declared_vector(box: Any) -> None:
    setup = set_fields_for_case(Path("."))

    apply_set_fields(box, setup.defaults, setup.regions)

    velocity = np.asarray(pyf.volVectorField.read_field(box, "U", False).internalField())
    expected = np.where(_IN_REGION[:, None], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0])
    assert_allclose(velocity, expected, err_msg="cases/box: U")


def test_the_written_file_reads_back_with_the_values_that_were_set(box: Any) -> None:
    setup = set_fields_for_case(Path("."))
    apply_set_fields(box, setup.defaults, setup.regions)

    reread = pyf.volScalarField.read_field(box, "alpha.water", False)

    assert_allclose(
        np.asarray(reread.internalField()),
        np.where(_IN_REGION, 1.0, 0.0),
        err_msg="cases/box: alpha.water re-read from 0/alpha.water",
    )


def test_every_named_field_is_reported_as_written(box: Any) -> None:
    setup = set_fields_for_case(Path("."))

    assert apply_set_fields(box, setup.defaults, setup.regions) == ["alpha.water", "U"]


def test_a_scalar_value_on_a_vector_field_names_the_field(box: Any) -> None:
    with pytest.raises(ValueError, match=r"field 'U' is a volVectorField on disk"):
        apply_set_fields(box, {"U": 1.0}, [])


def test_a_vector_value_on_a_scalar_field_names_the_field(box: Any) -> None:
    with pytest.raises(ValueError, match=r"field 'alpha.water' is a volScalarField on disk"):
        apply_set_fields(box, {"alpha.water": (1.0, 0.0, 0.0)}, [])


def test_a_field_missing_from_the_time_directory_names_it(box: Any) -> None:
    with pytest.raises(ValueError, match=r"field 'p_rgh' is not in 0/"):
        apply_set_fields(box, {"p_rgh": 0.0}, [])


def test_a_later_region_overrides_an_earlier_one_where_they_overlap(box: Any) -> None:
    setup = set_fields_for_case(CASES / "overlapping")

    apply_set_fields(box, setup.defaults, setup.regions)

    alpha = np.asarray(pyf.volScalarField.read_field(box, "alpha.water", False).internalField())
    assert_allclose(
        alpha, np.where(_IN_REGION, 1.0, 0.25), err_msg="cases/overlapping: alpha.water"
    )


def test_a_field_named_only_in_a_region_keeps_its_value_elsewhere(box: Any) -> None:
    setup = set_fields_for_case(CASES / "region_only")

    apply_set_fields(box, setup.defaults, setup.regions)

    alpha = np.asarray(pyf.volScalarField.read_field(box, "alpha.water", False).internalField())
    # 0.5 is what cases/box has on disk: with no default, the cells outside the
    # region are not written at all.
    assert_allclose(alpha, np.where(_IN_REGION, 1.0, 0.5), err_msg="cases/region_only: alpha.water")


def test_one_field_set_to_both_shapes_is_refused() -> None:
    # No mesh needed: the disagreement is in the declaration, so it is caught
    # before anything is read. CellGeometry only calls C() on what it is handed.
    mesh = SimpleNamespace(C=lambda: SimpleNamespace(internalField=lambda: np.zeros((4, 3))))
    region = Box(min=(0, 0, 0), max=(1, 1, 1))

    with pytest.raises(ValueError, match=r"field 'alpha.water' is set to values of"):
        apply_set_fields(mesh, {"alpha.water": 0.0}, [(region, {"alpha.water": (1.0, 0.0, 0.0)})])


@pytest.mark.parametrize(
    "value",
    [(1.0, 0.0), np.zeros(4), np.zeros((2, 3))],
    ids=["two_tuple", "four_vector", "matrix"],
)
def test_a_value_that_is_neither_scalar_nor_three_vector_is_refused(value: Any) -> None:
    # The script front door takes whatever Python it is handed, so the shape
    # check is here rather than in pydantic; no mesh is read.
    mesh = SimpleNamespace(C=lambda: SimpleNamespace(internalField=lambda: np.zeros((4, 3))))

    with pytest.raises(ValueError, match=r"field 'alpha.water' is set to a value of"):
        apply_set_fields(mesh, {"alpha.water": value}, [])


def test_a_numpy_scalar_sets_a_scalar_field(box: Any) -> None:
    # A script computes its value with numpy; np.float32 has no len().
    apply_set_fields(box, {"alpha.water": np.float32(0.25)}, [])

    alpha = np.asarray(pyf.volScalarField.read_field(box, "alpha.water", False).internalField())
    assert_allclose(alpha, np.full(16, 0.25), err_msg="cases/box: alpha.water from a numpy scalar")
