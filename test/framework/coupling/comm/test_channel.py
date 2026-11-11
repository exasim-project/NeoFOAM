# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The tags of the coupling's messages."""

from neofoam.framework.coupling.comm.channel import Tag


def test_the_tags_of_the_coupling_messages() -> None:
    assert Tag.data("x", "T_sink,h") == "data:x/T_sink,h"
    assert Tag.side("x") == "side:x"
    assert Tag.node("outer", "decision") == "outer/decision"
    assert Tag.TIME_DT == "time/dt"
