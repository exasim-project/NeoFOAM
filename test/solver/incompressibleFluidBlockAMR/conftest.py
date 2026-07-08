# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Fixtures for the incompressibleFluidBlockAMR solver tests.

* ``blockamr_session`` — initialize/finalize AMReX once for the whole session
  (AMReX may only be initialized once per process). ``run()`` detects this
  active session and reuses it instead of opening a nested one.
* ``box_case`` — copy the bundled periodic-box smoke case to a tmp dir and
  ``chdir`` into it, so config loading (``Path(".")``) and plotfile output land
  in an isolated directory.
"""

import os
import shutil
from pathlib import Path

import pytest

# Gate the whole package on the native engine being importable.
pytest.importorskip("neon")
import neon.blockamr as blockamr  # noqa: E402

os.environ.setdefault("AMREX_THE_ARENA_INIT_SIZE", "0")

CASE_SRC = Path(__file__).parent / "cases" / "box"
CYLINDER_CASE_SRC = Path(__file__).parent / "cases" / "cylinder"


@pytest.fixture(scope="session", autouse=True)
def blockamr_session():
    """Initialize AMReX once for all tests in this directory; do NOT finalize.

    AMReX and NeoN are not designed to share a process: NeoN pulls in a
    Kokkos-CUDA runtime (and JAX brings its own), and all three fight over the
    single CUDA primary context. ``amrex::Finalize`` frees the AMReX *arena
    allocator*'s device memory; when JAX/Kokkos have already dropped the context,
    that free aborts with ``CUDA error 709: context is destroyed``. Since
    ``neofoam`` is always imported here, we open the runtime but deliberately
    never run its finalizing ``__exit__`` — the OS reclaims GPU memory at process
    exit. (Standalone ``neon.blockamr`` still finalizes normally.)
    """
    blockamr.runtime().__enter__()
    yield


@pytest.fixture
def box_case(tmp_path, monkeypatch):
    """Copy the bundled periodic-box case to a tmp dir and chdir into it."""
    dst = tmp_path / "box"
    shutil.copytree(CASE_SRC, dst)
    monkeypatch.chdir(dst)
    return dst


@pytest.fixture
def cylinder_case(tmp_path, monkeypatch):
    """Copy the bundled non-periodic cylinder case to a tmp dir and chdir in."""
    dst = tmp_path / "cylinder"
    shutil.copytree(CYLINDER_CASE_SRC, dst)
    monkeypatch.chdir(dst)
    return dst
