# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Spec for where the solver graph puts the transport/turbulence correction.

``pimpleFoam.C`` runs ``laminarTransport.correct(); turbulence->correct();``
*inside* the outer corrector, under ``if (pimple.turbCorr())`` — so with the
native default (``turbOnFinalIterOnly true``) it fires once per time step, on
the final outer iteration, and with ``turbOnFinalIterOnly no`` once per outer
iteration. ``simpleFoam`` has no such gate: its loop body already runs once per
iteration, so a case that built a ``SimpleControl`` (no ``pimple_control``
model) must run the correction unconditionally.

These are unit tests on the gate wrapper the execution graph applies; the
end-to-end ordering is covered by the drop-in verification study.
"""

from neofoam.algorithms.solution_loop.control import PimpleControl, SimpleControl
from neofoam.framework.context import Context
from neofoam.framework.operations import Operation, SequentialOp
from neofoam.framework.types import OperationMetadata
from neofoam.solver.incompressibleFluid.incompressibleFluid import _under_turb_corr


def _counting_op(calls: list[str]) -> Operation:
    return Operation(
        func=SequentialOp(lambda _ctx: calls.append("correct")),
        metadata=OperationMetadata(op_name="of_correct_turbulence"),
    )


def test_correction_runs_only_on_the_outer_iterations_turb_corr_opens() -> None:
    control = PimpleControl(
        nOuterCorrectors=3, nCorrectors=1, momentumPredictor=True, turbCorr=True
    )
    calls: list[str] = []
    ctx = Context(fields={}, models={"pimple_control": control})

    gated = _under_turb_corr(_counting_op(calls))
    while control.loop():
        gated.run(ctx)

    # turbOnFinalIterOnly defaults to true: the final outer iteration only
    assert calls == ["correct"]


def test_correction_runs_every_outer_iteration_when_turb_on_final_iter_only_is_off() -> None:
    control = PimpleControl(
        nOuterCorrectors=3,
        nCorrectors=1,
        momentumPredictor=True,
        turbCorr=True,
        turbOnFinalIterOnly=False,
    )
    calls: list[str] = []
    ctx = Context(fields={}, models={"pimple_control": control})

    gated = _under_turb_corr(_counting_op(calls))
    while control.loop():
        gated.run(ctx)

    assert calls == ["correct"] * 3


def test_correction_is_ungated_for_a_simple_case() -> None:
    control = SimpleControl(momentumPredictor=True)
    calls: list[str] = []
    ctx = Context(fields={}, models={"simple_control": control})

    gated = _under_turb_corr(_counting_op(calls))
    while control.loop():
        gated.run(ctx)

    assert calls == ["correct"]


def test_gate_keeps_the_wrapped_operation_name() -> None:
    # the resolver places model ops by name; wrapping must stay invisible to it
    gated = _under_turb_corr(_counting_op([]))
    assert gated.operation_name == "of_correct_turbulence"
