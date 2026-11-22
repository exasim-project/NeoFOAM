# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The nanobind pin of ``pyproject.toml``.

The bindings resolve pybFoam's ``libnanobind.so`` at run time, so they need the ABI
pybFoam was built with. A build with isolation reads ``[build-system]``, one without
(``uv``, the ``dev`` extra) the environment: two places that must not drift apart.
"""

from __future__ import annotations

from pathlib import Path

import pytest

tomllib = pytest.importorskip("tomllib")  # standard library from Python 3.11 on

_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _nanobind(requirements: list[str]) -> list[str]:
    return [r for r in requirements if r.startswith("nanobind")]


def test_nanobind_is_pinned_the_same_for_isolated_and_in_env_builds():
    project = tomllib.loads(_PYPROJECT.read_text())

    isolated = _nanobind(project["build-system"]["requires"])
    in_env = _nanobind(project["project"]["optional-dependencies"]["dev"])

    assert isolated == in_env == ["nanobind>=3.0.1,<3.1"]
