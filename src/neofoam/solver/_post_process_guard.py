# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The serial-only guard every solver that wires ``postProcess`` applies."""

import pybFoam as pyf

from neofoam.postprocess import TableSet


def _refuse_parallel_post_processing(tables: TableSet) -> None:
    """Reject a ``-parallel`` run that declares tables (step 1 is serial-only).

    The aggregators sum over the local cells only, so a decomposed run would
    write per-rank partial values. The guard sits at the solver seam so the
    case fails at LOAD, before any solve time is spent, and because it covers
    every table: a table that never reduces (``rows``, ``residuals``) reaches
    no reduction to raise from and would otherwise write the master rank's
    share of the data without a word.
    """
    if tables.tables and pyf.Pstream.parRun():
        raise NotImplementedError(
            "postProcess is serial-only: the tables "
            f"{[t.name for t in tables.tables]} cannot be evaluated in a -parallel "
            "run because the aggregators have no MPI reduction yet. Remove "
            "system/postProcess.yaml / system/postProcess.py, or run serially."
        )
