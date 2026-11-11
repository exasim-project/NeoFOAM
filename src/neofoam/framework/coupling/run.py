# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The regions of a coupled case run, each on a thread of its own: :func:`run_participants`."""

from __future__ import annotations

import contextvars
import threading
from typing import Callable, Mapping, TypeVar

from .comm.channel import CommChannel, CommChannelClosed

T = TypeVar("T")


def run_participants(
    tasks: Mapping[str, Callable[[], T]], comm_channel: CommChannel
) -> dict[str, T]:
    """Run every region's task on its own thread; each task's result, by region.

    The threads report to the calling thread's execution observer. A failing
    region aborts the comm channel for all, so none waits forever; then the first
    error that is not a :class:`CommChannelClosed` (a consequence) is raised.
    """
    results: dict[str, T] = {}
    errors: list[BaseException] = []
    lock = threading.Lock()

    def run(region: str, task: Callable[[], T]) -> None:
        try:
            results[region] = task()
        except BaseException as error:  # re-raised in the calling thread
            with lock:
                errors.append(error)
            comm_channel.abort(tasks)
        finally:
            comm_channel.close(region)

    threads = [
        threading.Thread(
            target=contextvars.copy_context().run, args=(run, region, task), name=region
        )
        for region, task in tasks.items()
    ]
    comm_channel.start(tasks)
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise next((e for e in errors if not isinstance(e, CommChannelClosed)), errors[0])
    return {region: results[region] for region in tasks}
