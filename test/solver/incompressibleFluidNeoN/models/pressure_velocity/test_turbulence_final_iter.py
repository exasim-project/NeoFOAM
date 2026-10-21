# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Which iteration state the two algorithms hand the turbulence correction.

OpenFOAM's ``fvMatrix::solve()`` picks ``solvers/<field>Final`` and the ``*Final``
equation relaxation through ``psi.select(mesh.data().isFinalIteration())``.
pimpleFoam sets that flag on its final outer pass — which, under the default
``turbOnFinalIterOnly``, is exactly where ``turbulence->correct()`` runs — so the
k / epsilon / omega / nuTilda transports solve with the ``Final`` settings.
simpleFoam has no final outer iteration and never sets it.

The NeoN mirror carries the flag as a keyword on
:meth:`~neofoam.turbulence.native.NeoNHandle.correct`, which publishes it as the
``final_iter`` model the transport ``@operation``\\ s inject. The PIMPLE control
cannot be queried for it at this point: ``PimpleControl::loop`` zeroes its
corrector count on the pass that ends the loop, and ``turbulence_correct`` is
stepped after that — so the position in the graph is what fixes the value.

Losing this diverges ``pimpleFoam/RAS/pitzDaily``: k and epsilon stay on the base
``relTol 0.1`` for every solve, and the under-converged transports blow up into a
SIGFPE inside the pressure solve.

Both ``turbulence_correct`` operations are plain functions of their injected
arguments, so a recording double stands in for the handle — no case, no run.
"""

from __future__ import annotations

from typing import Any

from neofoam.solver.incompressibleFluidNeoN.models.pressure_velocity.pimpleAlgorithm import (
    turbulence_correct as pimple_turbulence_correct,
)
from neofoam.solver.incompressibleFluidNeoN.models.pressure_velocity.simpleAlgorithm import (
    turbulence_correct as simple_turbulence_correct,
)
from neofoam.turbulence.native import NeoNHandle


class _RecordingHandle:
    """Stands in for :class:`NeoNHandle`, recording the ``correct`` call it gets."""

    def __init__(self) -> None:
        self.final_iter: Any = "not called"

    def correct(self, U: Any, phi: Any, runtime: Any, final_iter: bool = False) -> None:
        self.final_iter = final_iter


def test_pimple_corrects_turbulence_as_the_final_outer_iteration() -> None:
    turbulence = _RecordingHandle()

    pimple_turbulence_correct(U="U", phi="phi", turbulence=turbulence, neon_runtime="rt")

    assert turbulence.final_iter is True


def test_simple_corrects_turbulence_outside_any_final_iteration() -> None:
    turbulence = _RecordingHandle()

    simple_turbulence_correct(U="U", phi="phi", turbulence=turbulence, neon_runtime="rt")

    assert turbulence.final_iter is False


def test_native_handle_publishes_final_iter_to_the_operation_context() -> None:
    """``correct`` puts the flag where the transport ``@operation``\\ s inject it."""
    handle = NeoNHandle(runtime=_NoOperations(), neon_runtime="rt", nu="nu")
    handle.validate(U="U")
    assert handle._ctx is not None
    assert handle._ctx.models["final_iter"] is False  # seeded by validate()

    handle.correct("U", "phi", "rt", final_iter=True)

    assert handle._ctx.models["final_iter"] is True


class _NoOperations:
    """A :class:`~neofoam.framework.model.ModelRuntime` with nothing to build or step."""

    def run_build(self) -> list[Any]:
        return []

    def native_operations(self) -> list[Any]:
        return []
