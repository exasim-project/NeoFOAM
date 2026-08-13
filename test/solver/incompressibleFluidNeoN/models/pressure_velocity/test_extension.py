# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The NeoN pressure-velocity seam is inert without an active contributor.

This is the property that made the seam safe to add to the shared NeoN
algorithms: a case that activates no rotating-frame model must assemble exactly
the equations ``simpleAlgorithm`` / ``pimpleAlgorithm`` wrote before the hooks
existed. For ``terms`` that means an ``Expression`` carrying no operator at all
(``size() == 0``), which is what makes ``+ ext.terms(U)`` a bit-identical
addition to the momentum sum; for ``predicted_flux`` it means the very flux
object the algorithm passed in (asserted by identity, not equality).

The fields here are real ``VolumeField``s on a single-cell mesh, not stand-ins:
``zero_expression`` reaches through the field to an executor, and which accessor
it reaches through decides whether the call can return to Python at all.

Contributions *are* registered on these hooks (``models.mrf``); they are skipped
here because no ``ModelRuntime`` is active on the Context, which is the mechanism
a case without ``constant/MRFProperties`` relies on.
"""

# NOTE: no `from __future__ import annotations` — the hooks are used as live
# ``Annotated[...]`` metadata elsewhere; keep this module PEP 563-free too.

from typing import Any

import neon._neon as nn

from neofoam.framework.context import Context
from neofoam.framework.model import Extension
from neofoam.solver.incompressibleFluidNeoN.models.pressure_velocity.extension import (
    momentum_extension,
    pressure_extension,
    zero_expression,
)


def single_cell_fields() -> tuple[Any, Any]:
    """A vector and a scalar ``VolumeField``, one cell, serial executor."""
    executor = nn.SerialExecutor()
    mesh = nn.create_single_cell_mesh(executor)
    return nn.VectorVolumeField(executor, "U", mesh), nn.ScalarVolumeField(executor, "p", mesh)


def bound_without_contributor(extension: Extension) -> Any:
    """*extension* bound to a Context on which no model is active."""
    return extension.resolve(Context(fields={}, models={}))


def test_momentum_terms_is_an_expression_with_no_operator() -> None:
    U, _ = single_cell_fields()

    folded = bound_without_contributor(momentum_extension).terms(U)

    assert folded.size() == 0


def test_zero_expression_matches_the_value_type_of_the_field() -> None:
    U, p = single_cell_fields()

    assert isinstance(zero_expression(U), nn.ExpressionVector)
    assert isinstance(zero_expression(p), nn.ExpressionScalar)


def test_momentum_constrain_touches_nothing() -> None:
    ran = bound_without_contributor(momentum_extension).constrain("U")

    assert ran == []


def test_predicted_flux_hands_back_the_flux_it_was_given() -> None:
    phiHbyA = object()

    predicted = bound_without_contributor(pressure_extension).predicted_flux(phiHbyA)

    assert predicted is phiHbyA


def test_pressure_constrain_corrected_velocity_touches_nothing() -> None:
    ran = bound_without_contributor(pressure_extension).constrain_corrected_velocity("U")

    assert ran == []
