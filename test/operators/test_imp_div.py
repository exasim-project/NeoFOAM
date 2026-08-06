# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Cross-backend parity of the implicit convection operators.

Both backends assemble the implicit operator and apply the matrix to the
current field: pybFoam builds the ``fvm.div`` matrix and returns ``M & psi``;
neon assembles the linear system of ``nn.imp.div`` and returns
``(A·psi - b) / V``. In exact arithmetic both equal the explicit divergence.

Named ``scheme`` ids hand the scheme to each backend as a string/TokenList.
The ``fromFvSchemes`` id passes no scheme at all, so each backend instead
selects ``div(phi,<field>)`` from the case's own ``system/fvSchemes`` — on the
neon side through the dictionary ``map_fv_schemes`` returned. That is the only
parameter exercising the dictionary-resolution path, and since ``div(phi,U)``
in the case carries the ``bounded`` convection prefix, it covers that prefix
reaching the assembled matrix. Note it compares the two backends against each
other, so it cannot itself pin the entry to stay ``bounded`` — weakening it
weakens both sides equally. The two tests below supply what it cannot:
``..._is_the_bounded_variant`` holds the case entry against a pybFoam-only
oracle, and ``..._keeps_bounded_div_prefix`` reads the mapped token list back
out of the dictionary with no mesh, case or executor involved.
"""

from __future__ import annotations

import neon._neon as nn
import numpy as np
import pytest
from backends import nb, pyb
from case_setup import flux, simulation
from conftest import EXECUTORS, MESH_NAMES

from neofoam import neofoam_bindings as nfb

# ids for the ``scheme`` parametrization; ``None`` means "no scheme string",
# i.e. let each backend look ``div(phi,<field>)`` up in system/fvSchemes.
DIV_SCHEME_IDS = [
    "linear",
    "upwind",
    "linearUpwind",
    "boundedUpwind",
    pytest.param(None, id="fromFvSchemes"),
]


@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("mesh", MESH_NAMES)
@pytest.mark.parametrize("scheme", DIV_SCHEME_IDS)
def test_imp_div_phi_T(mesh: str, scheme: str | None, executor: str) -> None:
    sim = simulation(mesh, executor)
    x, y, z = sim.mesh.cell_centres.T

    T = sim.field("T")
    T[:] = 2.0 + np.sin(np.pi * x) * np.cos(np.pi * y) + 0.3 * z

    U = sim.field("U")
    u = np.asarray(U)
    u[:, 0] = 1.0 + np.sin(np.pi * y) + 0.5 * np.sin(np.pi * x)
    u[:, 1] = 0.5 + np.cos(np.pi * x) + 0.5 * np.cos(np.pi * y)
    u[:, 2] = 0.1 + 0.2 * np.sin(np.pi * z)
    U[:] = u
    phi = flux(U)

    pyb_res = pyb.fvm.div(phi, T, scheme=scheme)
    nb_res = nb.imp.div(phi, T, scheme=scheme)

    rtol = 1e-9 if executor == "Serial" else 1e-8
    np.testing.assert_allclose(nb_res, pyb_res, rtol=rtol, atol=1e-12 * np.abs(pyb_res).max())


@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("mesh", MESH_NAMES)
@pytest.mark.parametrize("scheme", DIV_SCHEME_IDS)
def test_imp_div_phi_U(mesh: str, scheme: str | None, executor: str) -> None:
    sim = simulation(mesh, executor)
    x, y, z = sim.mesh.cell_centres.T

    U = sim.field("U")
    u = np.asarray(U)
    u[:, 0] = 1.0 + np.sin(np.pi * y) + 0.5 * np.sin(np.pi * x)
    u[:, 1] = 0.5 + np.cos(np.pi * x) + 0.5 * np.cos(np.pi * y)
    u[:, 2] = 0.1 + 0.2 * np.sin(np.pi * z)
    U[:] = u
    phi = flux(U)

    pyb_res = pyb.fvm.div(phi, U, scheme=scheme)
    nb_res = nb.imp.div(phi, U, scheme=scheme)

    rtol = 1e-9 if executor == "Serial" else 1e-8
    np.testing.assert_allclose(nb_res, pyb_res, rtol=rtol, atol=1e-12 * np.abs(pyb_res).max())


def test_fv_schemes_div_phi_U_is_the_bounded_variant() -> None:
    """Pin the case entry against an oracle only one backend can move.

    The ``fromFvSchemes`` parameter above compares the two backends against
    each other, so weakening ``div(phi,U)`` weakens both sides and it still
    passes. Here pybFoam resolving the entry is compared to pybFoam told
    ``bounded Gauss upwind`` outright, and separated from the unbounded
    scheme — so the prefix has to survive in the case file to stay green.
    """
    sim = simulation("cartesian_nx20", "Serial")
    x, y, z = sim.mesh.cell_centres.T

    U = sim.field("U")
    u = np.asarray(U)
    u[:, 0] = 1.0 + np.sin(np.pi * y) + 0.5 * np.sin(np.pi * x)
    u[:, 1] = 0.5 + np.cos(np.pi * x) + 0.5 * np.cos(np.pi * y)
    u[:, 2] = 0.1 + 0.2 * np.sin(np.pi * z)
    U[:] = u
    phi = flux(U)

    from_case = pyb.fvm.div(phi, U, scheme=None)

    np.testing.assert_allclose(from_case, pyb.fvm.div(phi, U, scheme="boundedUpwind"), rtol=1e-12)
    # The test flux is not divergence free, so the bounded correction is nonzero.
    assert not np.allclose(from_case, pyb.fvm.div(phi, U, scheme="upwind"))


def test_map_fv_schemes_keeps_bounded_div_prefix() -> None:
    """``map_fv_schemes`` round-trips ``bounded Gauss upwind`` token for token.

    Pure dictionary plumbing — no mesh, no case, no ``nn.initialize()``. This
    proves only that the mapping preserves the entry; that the div operator
    then resolves and applies it is what the ``fromFvSchemes`` parameter of
    the parity tests above covers.
    """
    div_schemes = nn.Dictionary()
    div_schemes.insert_token_list("div(phi,U)", nn.TokenList(["bounded", "Gauss", "upwind"]))
    fv_schemes = nn.Dictionary()
    fv_schemes.insert_dict("divSchemes", div_schemes)

    mapped = nfb.map_fv_schemes(fv_schemes)

    tokens = mapped.subDict("divSchemes").get_token_list("div(phi,U)")
    assert [tokens.get_string(i) for i in range(tokens.size())] == ["bounded", "Gauss", "upwind"]
