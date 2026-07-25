# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Spec for the VoF PIMPLE inner-loop predicate: finalIteration flag on the mesh.

OpenFOAM's ``pimpleControl::loop()`` sets the mesh's final-iteration state so
``fvMatrix::solve`` picks the ``<field>Final`` solver settings on the last outer
corrector, and — crucially — calls ``setFinalIteration(false)`` when the loop
*ends*, before function objects execute (``pimpleControl.C:222``).

The VoF graph runs ``turbulence_correction`` *inside* the inner loop and
``write_output`` (which triggers equation-solving function objects, e.g.
``electricPotential``) *after* it. So unlike the incompressibleFluid graph (where
turbulence runs after the loop and the flag must stay raised), the VoF predicate
must LOWER the flag on loop exit — otherwise a function object solve picks the
non-existent ``<field>Final`` sub-dict and dies.
"""

from neofoam.algorithms.solution_loop.control import PimpleControl  # noqa: E402
from neofoam.framework.context import Context  # noqa: E402
from neofoam.solver.incompressibleVoF.models.pressure_velocity.pimpleAlgorithm import (  # noqa: E402
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


def test_pimple_mode_marks_only_last_outer_iteration_final() -> None:
    pimple = PimpleControl(nCorrectors=2, nOuterCorrectors=3, momentumPredictor=True)
    mesh = MeshSpy()
    ctx = _ctx(pimple, mesh)

    flags = []
    while inner_loop(ctx):
        flags.append(mesh.final)

    assert flags == [False, False, True]


def test_flag_lowered_when_outer_loop_exits() -> None:
    # Mirrors pimpleControl::loop()'s setFinalIteration(false) on exit: the
    # trailing write_output/function-objects must see the base solver dict.
    pimple = PimpleControl(nCorrectors=2, momentumPredictor=True)  # nOuter = 1
    mesh = MeshSpy()
    ctx = _ctx(pimple, mesh)

    assert inner_loop(ctx) is True
    assert mesh.final is True  # inside the final outer iteration
    assert inner_loop(ctx) is False  # loop exit
    assert mesh.final is False  # lowered before function objects run
