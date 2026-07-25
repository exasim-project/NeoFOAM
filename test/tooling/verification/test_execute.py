# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""``failure_reason`` picks the most explanatory line out of a failed log.

Pins the ``-postProcess`` branch: it must win over the generic tail fallback
(and be returned verbatim, not truncated/reformatted) so
``runner.py::_decide`` can match it exactly and classify the run as
UNSUPPORTED_CASE rather than a solver crash.
"""

from neofoam.tooling.verification.execute import (
    POSTPROCESS_NOT_IMPLEMENTED,
    failure_reason,
)


def test_failure_reason_detects_postprocess_not_implemented() -> None:
    log = "Using: OpenFOAM-v2406\n\nneofoam: solver -postProcess mode not implemented\n"

    assert failure_reason(log) == POSTPROCESS_NOT_IMPLEMENTED


def test_failure_reason_falls_back_to_log_tail_otherwise() -> None:
    log = "Starting time loop\nsome other unrelated crash\n"

    assert failure_reason(log) == "Starting time loop some other unrelated crash"
