# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Spec for the PIMPLE inner-loop predicate: finalIteration flag on the mesh.

OpenFOAM's ``pimpleControl::loop()`` marks the mesh's final-iteration state
so ``fvMatrix::solve`` picks the ``<field>Final`` solver settings (relTol 0)
on the last outer corrector — in PISO mode (one outer corrector) on every
solve. The framework's ``inner_loop`` predicate must mirror that, and must
leave the flag raised after the loop exits: the turbulence correction runs
*after* the inner loop in the framework graph but *inside* the final outer
iteration natively.
"""

from neofoam.algorithms.solution_loop.control import PimpleControl  # noqa: E402
from neofoam.framework.context import Context  # noqa: E402
from neofoam.solver.incompressibleFluid.models.pressure_velocity.pimpleAlgorithm import (  # noqa: E402
    inner_loop,
)


class MeshSpy:
    """Records setFinalIteration calls the way pybFoam's fvMesh receives them."""

    def __init__(self) -> None:
        self.final = False
        self.calls: list[bool] = []

    def setFinalIteration(self, on: bool) -> None:
        self.final = on
        self.calls.append(on)


def _ctx(pimple: PimpleControl, mesh: MeshSpy) -> Context:
    return Context(fields={}, models={"pimple_control": pimple}, mesh=mesh)


def test_piso_mode_marks_every_iteration_final() -> None:
    pimple = PimpleControl(nCorrectors=2, momentumPredictor=True)  # nOuter = 1
    mesh = MeshSpy()
    ctx = _ctx(pimple, mesh)

    flags = []
    while inner_loop(ctx):
        flags.append(mesh.final)

    assert flags == [True]
    assert mesh.final is True  # stays raised for the trailing turbulence solves


def test_pimple_mode_marks_only_last_outer_iteration_final() -> None:
    pimple = PimpleControl(nCorrectors=2, nOuterCorrectors=3, momentumPredictor=True)
    mesh = MeshSpy()
    ctx = _ctx(pimple, mesh)

    flags = []
    while inner_loop(ctx):
        flags.append(mesh.final)

    assert flags == [False, False, True]
    assert mesh.final is True


def test_flag_drops_again_on_the_next_time_step() -> None:
    pimple = PimpleControl(nCorrectors=2, nOuterCorrectors=2, momentumPredictor=True)
    mesh = MeshSpy()
    ctx = _ctx(pimple, mesh)

    while inner_loop(ctx):
        pass
    assert mesh.final is True

    # next time step: first outer iteration is non-final again
    assert inner_loop(ctx) is True
    assert mesh.final is False


def test_exit_call_does_not_touch_the_flag() -> None:
    pimple = PimpleControl(nCorrectors=2, momentumPredictor=True)
    mesh = MeshSpy()
    ctx = _ctx(pimple, mesh)

    assert inner_loop(ctx) is True
    calls_while_looping = len(mesh.calls)
    assert inner_loop(ctx) is False  # loop exit
    assert len(mesh.calls) == calls_while_looping
