# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""``regions_recorder``: a test in-situ type that records what a multiRun hands it.

Each call appends the regions it was handed with their time index and mean ``T``
(a region on another process is ``None``), and the drawing thread, to
:data:`CALLS`: a module global, because the run draws a private copy of the entry.
"""

import threading
from typing import Any, Literal, Optional

from neofoam.framework.context import Context
from neofoam.postprocess.insitu import InSitu

#: Per call: region -> (time index, mean T), or None for another process's region; the thread.
CALLS: list[Any] = []


@InSitu.register
class RegionsRecorder(InSitu):
    type: Literal["regions_recorder"] = "regions_recorder"
    fail: bool = False

    def execute_regions(self, regions: dict[str, Optional[Context]]) -> None:
        if self.fail:
            raise RuntimeError("the renderer broke")
        CALLS.append(
            (
                {
                    name: None if ctx is None else (ctx.time.index, float(ctx.fields["T"].mean()))
                    for name, ctx in regions.items()
                },
                threading.current_thread().name,
            )
        )

    def finalize(self) -> None:
        CALLS.append("finalize")
