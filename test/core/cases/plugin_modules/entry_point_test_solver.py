# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""A plugin solver that records the argv it is run with."""

calls: list[list[str]] = []


def run(argv: list[str]) -> None:
    """Record ``argv`` instead of running a case."""
    calls.append(argv)
