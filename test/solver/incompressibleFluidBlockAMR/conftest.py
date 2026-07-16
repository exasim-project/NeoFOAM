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
import blockamr  # noqa: E402

os.environ.setdefault("AMREX_THE_ARENA_INIT_SIZE", "0")
# Preallocate the JAX/XLA pool up front (a fixed fraction) — much faster than
# on-demand growth, which pays an allocation cost every step. The AMReX arena
# for these small test meshes is tiny (~0.1 GB, it grows on demand from init
# size 0), so the ~20 % left by MEM_FRACTION is ample; both allocators coexist
# on the one device.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.8")

CASE_SRC = Path(__file__).parent / "cases" / "box"
BOX_CPP_CASE_SRC = Path(__file__).parent / "cases" / "box_cpp"
CYLINDER_CASE_SRC = Path(__file__).parent / "cases" / "cylinder"
CAVITY_CASE_SRC = Path(__file__).parent / "cases" / "cavity"


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
    exit. (Standalone ``blockamr`` still finalizes normally.)
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
def box_cpp_case(tmp_path, monkeypatch):
    """Copy the periodic-box case whose ``solvers.U`` selects the cpp backend."""
    dst = tmp_path / "box_cpp"
    shutil.copytree(BOX_CPP_CASE_SRC, dst)
    monkeypatch.chdir(dst)
    return dst


@pytest.fixture
def cylinder_case(tmp_path, monkeypatch):
    """Copy the bundled non-periodic cylinder case to a tmp dir and chdir in."""
    dst = tmp_path / "cylinder"
    shutil.copytree(CYLINDER_CASE_SRC, dst)
    monkeypatch.chdir(dst)
    return dst


@pytest.fixture
def cavity_case(tmp_path, monkeypatch):
    """Copy the bundled lid-driven cavity case to a tmp dir and chdir in."""
    dst = tmp_path / "cavity"
    shutil.copytree(CAVITY_CASE_SRC, dst)
    monkeypatch.chdir(dst)
    return dst
