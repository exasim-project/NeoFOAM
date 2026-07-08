# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""INT-5 / C5 — OpenFOAM-style patch BCs map to neon.blockamr VectorBC.

Supported: inlet ``fixedValue`` (Dirichlet), outlet ``zeroGradient`` (Neumann),
wall ``noSlip``, and free-slip ``slip`` / ``symmetry`` / ``symmetryPlane``
(``SlipBC`` — no penetration + zero tangential shear).
"""

import pytest

pytest.importorskip("neon")

from neon.blockamr.bc import NeumannBC, SlipBC, VectorDirichletBC  # noqa: E402

from neofoam.solver.incompressibleFluidBlockAMR.models.bc_mapping import (  # noqa: E402
    build_vector_bc,
    map_patch,
)


def test_fixed_value_maps_to_dirichlet():
    bc = map_patch({"type": "fixedValue", "value": [1.0, 0.0, 0.0]})
    assert isinstance(bc, VectorDirichletBC)
    assert list(bc.vec) == pytest.approx([1.0, 0.0, 0.0])


def test_zero_gradient_maps_to_neumann():
    assert isinstance(map_patch({"type": "zeroGradient"}), NeumannBC)


def test_no_slip_maps_to_zero_dirichlet():
    bc = map_patch({"type": "noSlip"})
    assert isinstance(bc, VectorDirichletBC)
    assert list(bc.vec) == pytest.approx([0.0, 0.0, 0.0])


def test_build_vector_bc_places_faces():
    bc = build_vector_bc(
        {
            "xlo": {"type": "fixedValue", "value": [1.0, 0.0, 0.0]},  # inlet
            "xhi": {"type": "zeroGradient"},  # outlet
            "ylo": {"type": "noSlip"},  # wall
            "yhi": {"type": "noSlip"},  # wall
        }
    )
    assert isinstance(bc.lo[0], VectorDirichletBC)  # xlo inlet
    assert list(bc.lo[0].vec) == pytest.approx([1.0, 0.0, 0.0])
    assert isinstance(bc.hi[0], NeumannBC)  # xhi outlet
    assert isinstance(bc.lo[1], VectorDirichletBC)  # ylo wall (noSlip)


@pytest.mark.parametrize("bc_type", ["slip", "symmetry", "symmetryPlane"])
def test_slip_symmetry_maps_to_slip(bc_type):
    assert isinstance(map_patch({"type": bc_type}), SlipBC)


def test_unknown_face_rejected():
    with pytest.raises(ValueError):
        build_vector_bc({"north": {"type": "noSlip"}})
