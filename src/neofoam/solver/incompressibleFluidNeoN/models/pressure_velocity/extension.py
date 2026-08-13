# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The extension seams of the NeoN pressure-velocity operations.

The NeoN twin of
:mod:`neofoam.solver.incompressibleFluid.models.pressure_velocity.extension`:
a model hooks into the NeoN ``UEqn`` / ``pEqn`` port with
``@<model>.contributes(<extension>.<hook>)`` and the algorithms call each hook
once on the injected handle, so no operation has to know which models a case
activated. The hooks are separate from the pybFoam family's because they speak
NeoN types (``neon._neon`` fields, ``NeoN::dsl`` operators) rather than
``volVectorField`` / ``fvVectorMatrix``.

Each hook is named for the quantity it exposes and the moment it exposes it,
never for what a contributor does to it, and each picks one of the
:class:`~neofoam.framework.model.Kind` s. With no contributor every hook is the
identity: ``terms`` is an empty expression, ``predicted_flux`` hands back the
flux it was given and the two ``constrain`` hooks do nothing — the assembled
equations are the ones the algorithms wrote before this seam existed.

Example::

    @myModel.contributes(momentum_extension.terms)
    def my_momentum_term(U: Any, my_runtime: Annotated[Any, "models"]) -> Any:
        return nn.exp.source(my_runtime.acceleration(U))
"""

from typing import Any

import neon._neon as nn  # NeoN Python bindings

from neofoam.framework.model import Extension, Kind, fold

__all__ = [
    "momentum_extension",
    "pressure_extension",
    "zero_expression",
]

momentum_extension = Extension("momentum")
pressure_extension = Extension("pressure")


def zero_expression(field: Any) -> Any:
    """The additive identity of the NeoN DSL for *field*'s value type.

    An ``Expression`` built from an executor alone carries no operator, so
    adding it to an equation assembles a bit-identical matrix. Use it to seed a
    fold whose contributions are NeoN operators.
    """
    cls = nn.ExpressionVector if isinstance(field, nn.VectorVolumeField) else nn.ExpressionScalar
    # Via the internal vector: VolumeField::exec() returns a std::variant that
    # volumeField.cpp binds without <nanobind/stl/variant.h>, so it has no
    # return caster and raises. vectors.cpp includes the header; same executor.
    return cls(field.internal_vector().exec())


# ---------------------------------------------------------------------------
# Hooks — one @defines function per point the operations expose
# ---------------------------------------------------------------------------


@momentum_extension.defines
def terms(U: Any, contributions: list[Any]) -> Any:
    """Terms folded into the momentum sum at ``+ ext.terms(U)``.

    The seed is an empty ``nn.Expression`` — NeoN's additive identity, so the
    no-contributor path assembles the matrix the algorithm wrote before this
    seam existed, bit for bit. A contribution's operator joins with ``+``; a
    term that belongs on native's right-hand side (``== source``) returns
    ``negated(term)``.
    """
    return fold(zero_expression(U), contributions)


@momentum_extension.defines
def constrain(U: Any) -> None:
    """The velocity, before the momentum matrix is assembled from it.

    Native's ``fvConstraints().constrain(U)``: whatever a contribution writes
    into the boundary here is what the boundary coefficients of ``div(phi,U)``
    and ``laplacian(nuEff,U)`` are built from.
    """


@pressure_extension.defines(kind=Kind.PIPELINE)
def predicted_flux(phiHbyA: Any) -> Any:
    """The predicted flux the pressure equation will see, already summed.

    One point, because native does everything to this flux here:
    ``MRF.makeRelative(phi)``, ``fvc::makeRelative(phi, U)`` on a moving mesh,
    and ``adjustPhi`` when p needs a reference. Contributions apply in
    registration order, each transforming the previous one's output, so those
    compose the way native stacks the calls; a contribution returning ``None``
    passes the flux through untouched.
    """


@pressure_extension.defines
def constrain_corrected_velocity(U: Any) -> None:
    """The velocity, after the pressure corrector updated and corrected it.

    Native needs no such point: a ``fixedValue``/``noSlip`` patch keeps an
    assigned value through ``correctBoundaryConditions``. NeoN's fixed-value
    boundary re-derives its face values from the stored uniform on every
    correct, so a contribution that owns a boundary value has to re-apply it
    here to hold the state native holds.
    """
