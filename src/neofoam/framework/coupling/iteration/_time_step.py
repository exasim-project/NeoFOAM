# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The time-step agreement of a coupled group: the op ``<region>.time_step``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Optional

from neofoam.framework.operations import IterativeOp, Operation

from ..comm.channel import Tag

if TYPE_CHECKING:
    from neofoam.framework.context import Context

    from ..exchanges import ParticipantExchanges


SET_TIME_STEP = "set_time_step"


INCREMENT_TIME = "increment_time"


def operation_names(ops: Iterable[Operation]) -> set[str]:
    """The names of ``ops`` and of every operation inside their loops."""
    names: set[str] = set()
    for op in ops:
        if op.operation_name:
            names.add(op.operation_name)
        if isinstance(op.func, IterativeOp):
            names |= operation_names(op.sub_operations)
    return names


def serial_holder(ops: Iterable[Operation], name: str, serial: dict[str, str]) -> Optional[str]:
    """The node of the outermost loop of ``serial`` (by label) holding operation ``name``."""
    for op in ops:
        if isinstance(op.func, IterativeOp):
            if op.operation_name in serial and name in operation_names(op.sub_operations):
                return serial[op.operation_name]
            inside = serial_holder(op.sub_operations, name, serial)
            if inside is not None:
                return inside
    return None


class TimeStep:
    def __init__(self, coupling: ParticipantExchanges) -> None:
        self._coupling = coupling

    def __call__(self, ctx: Context) -> None:
        coupling = self._coupling
        region, leader, comm_channel = coupling.region, coupling.leader, coupling.comm_channel
        asked = ctx.time.delta_t
        if region != leader:
            comm_channel.send(region, leader, Tag.TIME_DT, asked)
            agreed = comm_channel.receive(leader, region, Tag.TIME_DT)
        else:
            agreed = min(
                asked if other == region else comm_channel.receive(other, region, Tag.TIME_DT)
                for other in coupling.group
            )
            for other in coupling.group:
                if other != region:
                    comm_channel.send(region, other, Tag.TIME_DT, agreed)
        ctx.time.delta_t = agreed
