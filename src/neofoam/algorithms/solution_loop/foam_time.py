# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""FoamTime — a LoopBackend mirroring a LoopState onto an OpenFOAM ``Foam::Time``.

The :class:`~neofoam.algorithms.solution_loop.solution_loop.SolutionLoop` owns the
advancement; this backend keeps a backend ``Time`` in lockstep so field IO lands
in the right time directory. It is **duck-typed** on the wrapped time object — it
only calls ``setDeltaT`` / ``increment`` — so it carries no pybFoam import and is
unit-testable with a fake. The solver constructs it with the real
``pybFoam.Time`` and injects it (see the solver's ``loop_backend_steps``).

``setTime`` can't be called from Python (``Foam::instant`` has no binding
constructor), so we push the (already-computed) ``deltaT`` and reconcile the
backend clock by ``increment``-ing it until its step index matches the state's.
It does **not** write — that is the ``fieldWriter`` Model's job.
"""

from __future__ import annotations

from typing import Any

from neofoam.algorithms.solution_loop.loop_state import LoopState


class FoamTime:
    """Mirror the Python :class:`LoopState` onto a backend ``Foam::Time`` clock."""

    def __init__(self, foam_time: Any) -> None:
        self._t = foam_time
        self._index = 0
        # Optional opaque keep-alive slot. The solver stashes the backend
        # ``argList`` here so it outlives the wrapped ``Foam::Time`` (which holds
        # it only by raw reference) — critical under ``-parallel``, where the
        # argList owns the MPI session. Never read; held purely for lifetime.
        self._foam_arglist: Any = None

    def update(self, state: LoopState) -> None:
        self._t.setDeltaT(state.delta_t)
        while self._index < state.index:
            self._t.increment()
            self._index += 1
