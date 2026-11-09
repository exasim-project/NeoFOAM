# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Every in-process comm channel of the multiRun tests pickles its messages."""

from collections.abc import Iterator

import pytest

from neofoam.framework.coupling import InProcessCommChannel


@pytest.fixture(scope="session", autouse=True)
def _pickled_messages() -> Iterator[None]:
    """A message that is not plain data fails (session-wide: module fixtures run multiRun)."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(InProcessCommChannel, "PICKLE", True)
        yield
