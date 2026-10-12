# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""A runtime's region selects the case files its solver reads.

The ``counter`` solver loads ``constant/increment.yaml`` for its runtime's region.
``cases/regions`` holds a different increment per region and another one at the
single-region path, so a runtime that read the wrong file shows up in its value;
a runtime without a region reads the single-region file, as before regions.
``finalize`` runs the solver's ``@finalizer`` on the region's context.
"""

from typing import Optional

import pytest

from framework.solver.region_counter import counter


@pytest.mark.parametrize(("region", "expected"), [("left", 2.0), ("right", 3.0), (None, 1.0)])
def test_runtime_region_selects_the_region_configs(region: Optional[str], expected: float) -> None:
    ctx = counter.instantiate(region=region).initialize()

    assert ctx.models["increment"] == expected


def test_runtime_finalize_runs_the_registered_finalizer() -> None:
    runtime = counter.instantiate(region="left")
    ctx = runtime.initialize()

    runtime.finalize(ctx)

    assert ctx.models["finalized"] is True
