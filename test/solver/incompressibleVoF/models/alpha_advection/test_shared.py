# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Spec for ``shared_field_build_steps`` — the fields every advection scheme owns.

Two levels, mirroring ``test/solver/incompressibleVoF/test_create_fields.py``:

* the **step list** (names, categories, dependency edges) is pure description
  and is asserted in-process;
* the **values** those steps produce come from the one executed pipeline of
  ``cases/vofRow4`` (see ``test/solver/incompressibleVoF/conftest.py`` — one
  ``Foam::Time`` per process, so it runs in a worker subprocess and dumps JSON).
  These steps are reached exactly as production reaches them: through
  ``create_init(...).run()`` with MULES selected.

Every expectation is hand-derived from that case, which is a unit cube cut into
4 cells along x (cell volume 0.25, x-face area 1, uniform spacing so the linear
interpolation weight of every internal face is exactly 0.5) with the interFoam
damBreak transport properties:

* ``0/alpha.water`` = ``(0 0.25 0.75 1)``, and ``alpha2 = 1 - alpha1``;
* ``rho1 = 1000``, ``rho2 = 1``, so
  ``rho = alpha1*1000 + alpha2*1 = 999*alpha1 + 1 = (1, 250.75, 750.25, 1000)``;
* ``0/U`` = ``(2 0 0)`` so ``phi = 2`` on every internal x-face, and
  ``rhoPhi = interpolate(rho)*phi = (rho_own + rho_nei)`` there —
  ``(251.75, 1001, 1750.25)``.

``rtol=1e-12`` allows for round-off in OpenFOAM's geometric interpolation
weights only; the arithmetic itself is exact in binary floating point.
"""

from __future__ import annotations

from numpy.testing import assert_allclose

from neofoam.solver.incompressibleVoF.models.alpha_advection.shared import (
    shared_field_build_steps,
)

from ...conftest import BuiltCase


def test_shared_steps_are_phi_mixture_alphas_rho_and_the_alpha_fluxes() -> None:
    steps = shared_field_build_steps()
    assert [(step.name, step.category) for step in steps] == [
        ("fields.phi", "fields"),
        ("models.mixture", "models"),
        ("fields.alpha1", "fields"),
        ("fields.alpha2", "fields"),
        ("fields.rho", "fields"),
        ("fields.rhoPhi", "fields"),
        ("fields.alphaPhiUn", "fields"),
        ("fields.alphaPhi10", "fields"),
    ]


def test_shared_steps_declare_the_edges_that_order_them() -> None:
    # phi from U, the mixture from U/phi/mesh, both phase fractions from the
    # mixture, rho from the mixture + both fractions, rhoPhi from rho + phi,
    # and the two createAlphaFluxes.H fluxes: alphaPhiUn (zero-init) from phi
    # alone, alphaPhi10 (phi*interpolate(alpha1)) from phi and alpha1.
    steps = shared_field_build_steps()
    assert [step.depends_on for step in steps] == [
        ["fields.U"],
        ["fields.U", "fields.phi", "mesh"],
        ["models.mixture"],
        ["models.mixture"],
        ["models.mixture", "fields.alpha1", "fields.alpha2"],
        ["fields.rho", "fields.phi"],
        ["fields.phi"],
        ["fields.phi", "fields.alpha1"],
    ]


def test_no_shared_field_is_flagged_for_the_python_field_writer() -> None:
    # VoF writes through OpenFOAM's AUTO_WRITE, so none of these steps opts into
    # ``Context.write_fields``.
    assert [step.write for step in shared_field_build_steps()] == [False] * 8


def test_phase_fractions_are_the_mixtures_own_fields(vof_row4: BuiltCase) -> None:
    # alpha1/alpha2 are not read again — they are the mixture's fields, named
    # after the case's ``phases (water air)``.
    assert vof_row4.result["mixture"]["alpha1_name"] == "alpha.water"
    assert vof_row4.result["mixture"]["alpha2_name"] == "alpha.air"


def test_alpha2_is_the_complement_of_alpha1(vof_row4: BuiltCase) -> None:
    # 0/alpha.water is (0 0.25 0.75 1).
    assert vof_row4.internal("alpha2") == [1.0, 0.75, 0.25, 0.0]


def test_mixture_densities_come_from_transport_properties(
    vof_row4: BuiltCase,
) -> None:
    # constant/transportProperties: water rho 1000, air rho 1.
    assert (
        vof_row4.result["mixture"]["rho1"],
        vof_row4.result["mixture"]["rho2"],
    ) == (1000.0, 1.0)


def test_rho_is_the_alpha_weighted_mixture_density(vof_row4: BuiltCase) -> None:
    # rho = alpha1*rho1 + alpha2*rho2 = 999*alpha1 + 1. Swapping rho1/rho2 would
    # give (1000, 750.25, 250.75, 1), so the weighting is pinned, not just the
    # end points.
    assert_allclose(
        vof_row4.internal("rho"),
        [1.0, 250.75, 750.25, 1000.0],
        rtol=1e-12,
        err_msg="vofRow4: rho must be alpha1*rho1 + alpha2*rho2",
    )


def test_rho_phi_is_the_interpolated_density_times_the_face_flux(
    vof_row4: BuiltCase,
) -> None:
    # rhoPhi = interpolate(rho)*phi on the 3 internal faces: the uniform mesh
    # halves the neighbouring rho values and phi = 2 doubles them again, so each
    # face carries rho_own + rho_nei.
    assert_allclose(
        vof_row4.internal("rhoPhi"),
        [251.75, 1001.0, 1750.25],
        rtol=1e-12,
        err_msg="vofRow4: rhoPhi must be fvc.interpolate(rho)*phi",
    )


def test_alpha_phi_un_is_registered_under_its_final_name(vof_row4: BuiltCase) -> None:
    # createAlphaFluxes.H: registered under the literal name "alphaPhiUn" so
    # e.g. the scalarTransport functionObject's ``phase`` lookup finds it.
    assert vof_row4.result["registered_names"]["alphaPhiUn"] == "alphaPhiUn"


def test_alpha_phi_un_is_zero_right_after_build(vof_row4: BuiltCase) -> None:
    # Zero-initialised (dimensionedScalar(phi.dimensions(), Zero)); MULES
    # fills it in place during the alpha solve, not at build time.
    assert vof_row4.internal("alphaPhiUn") == [0.0, 0.0, 0.0]
