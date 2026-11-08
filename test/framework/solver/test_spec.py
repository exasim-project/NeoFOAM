# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""A solver operation carries the fields it declares to solve (``solves=``).

``counter`` (``region_counter``) declares ``add_increment`` to solve ``total``.
"""

from framework.solver.region_counter import counter


def test_spec_operation_records_the_fields_it_solves() -> None:
    runtime = counter.instantiate(region="left")

    assert runtime.operations["add_increment"].metadata.solves == ["total"]
