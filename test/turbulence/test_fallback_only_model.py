# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The fallback-only model shape (``realizableKE``).

A fallback-only model registers a name (so selection and the MCP see it) but
declares **only** the pybFoam-OpenFOAM backend — no ``@build`` and no native
``@operation``, just one ``fallback=True`` ``correct``. It is usable only when a
solver selects ``fallback=True``; the native NeoN path raises cleanly.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from neofoam.turbulence import momentumTransportModel
from neofoam.turbulence.config import TurbulencePropertiesConfig
from neofoam.turbulence.fallback import FallbackHandle
from neofoam.turbulence.selection import select_turbulence_model

_TURBULENCE_PROPERTIES = """\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      turbulenceProperties;
}

simulationType      RAS;

RAS
{
    RASModel        realizableKE;
    turbulence      on;
    printCoeffs     on;
}
"""


@pytest.fixture
def case_dir(tmp_path: Path) -> Path:
    """A minimal case whose turbulenceProperties selects ``realizableKE``."""
    constant = tmp_path / "constant"
    constant.mkdir()
    (constant / "turbulenceProperties").write_text(_TURBULENCE_PROPERTIES)
    return tmp_path


def test_registered_in_the_family() -> None:
    assert "realizableKE" in momentumTransportModel.registered_names()


def test_fallback_true_returns_a_fallback_handle_with_one_op(case_dir: Path) -> None:
    cfg = TurbulencePropertiesConfig.load(case_dir=case_dir)
    handle = select_turbulence_model(
        cfg, fallback=True, case_dir=case_dir, of_factory=MagicMock()
    )
    assert isinstance(handle, FallbackHandle)
    op_names = [op.metadata.op_name for op in handle.operations]
    assert op_names == ["realizableKECorrect"]


def test_fallback_false_raises_no_native_closure(case_dir: Path) -> None:
    cfg = TurbulencePropertiesConfig.load(case_dir=case_dir)
    with pytest.raises(ValueError, match="no native NeoN closure"):
        select_turbulence_model(cfg, fallback=False, case_dir=case_dir)
