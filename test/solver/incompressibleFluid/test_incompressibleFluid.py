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

import pytest

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


# Each row is the control a case would build, and how many of its loop passes the
# gate must open on. The control is *described* here and constructed in the body:
# looping it consumes it, so an instance must not be shared across runs.
_PIMPLE = {"nOuterCorrectors": 3, "nCorrectors": 1, "momentumPredictor": True, "turbCorr": True}


@pytest.mark.parametrize(
    ("control_cls", "model_name", "control_kwargs", "expected_calls"),
    [
        # turbOnFinalIterOnly defaults to true: the final outer iteration only
        (PimpleControl, "pimple_control", _PIMPLE, 1),
        (PimpleControl, "pimple_control", {**_PIMPLE, "turbOnFinalIterOnly": False}, 3),
        # simpleFoam has no gate: its loop body already runs once per iteration
        (SimpleControl, "simple_control", {"momentumPredictor": True}, 1),
    ],
    ids=["pimple_final_iteration_only", "pimple_every_outer_iteration", "simple_ungated"],
)
def test_the_gate_opens_on_the_outer_iterations_turb_corr_opens(
    control_cls: type, model_name: str, control_kwargs: dict, expected_calls: int
) -> None:
    control = control_cls(**control_kwargs)
    calls: list[str] = []
    ctx = Context(fields={}, models={model_name: control})

    gated = _under_turb_corr(_counting_op(calls))
    while control.loop():
        gated.run(ctx)

    assert calls == ["correct"] * expected_calls


def test_gate_keeps_the_wrapped_operation_name() -> None:
    # the resolver places model ops by name; wrapping must stay invisible to it
    gated = _under_turb_corr(_counting_op([]))
    assert gated.operation_name == "of_correct_turbulence"
