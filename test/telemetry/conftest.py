# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

from typing import Iterator

import pytest

from neofoam import telemetry


@pytest.fixture(autouse=True)
def reset_telemetry() -> Iterator[None]:
    """Keep the module-level telemetry state isolated between tests."""
    telemetry.shutdown()
    yield
    telemetry.shutdown()
