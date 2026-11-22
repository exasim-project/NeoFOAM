# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Process-wide NeoN/Kokkos initialization: executor selection and teardown.

``requested_executor`` and the report wording are a config read
(:class:`~neofoam.solver.neon_runtime.NeoNControlConfig`) and string formatting,
tested in-process — so the wording holds on a host-only build too, where the
subprocess test below has nothing to provoke and skips. Both report tests run
from ``test/setup_pimple``, because the report names the executor and
``requested_executor`` reads it from the case's ``system/controlDict``. The
other two behaviors are only observable across a whole process life, hence
subprocess-driven:

**A failed GPU init must say what to do about it.** Kokkos brings up every
backend it was compiled with, so a CUDA-enabled build takes a CUDA context per
rank even for a Serial executor; when that fails Kokkos reports a bare CUDA
error code and a Kokkos source line, which explains neither why a Serial run
needs a device at all nor where to read up on it.
``KOKKOS_VISIBLE_DEVICES`` pointing at a device ordinal that cannot exist forces
exactly that failure without disturbing the real device, so the diagnostic can be
asserted on a machine whose GPU is healthy. A build without a GPU backend
initializes fine and has no diagnostic to show — that run is reported by the
worker and skipped.

**The finalize handler must not abort at interpreter teardown.**
``ensure_neon_initialized`` registers a finalize handler via ``atexit``. Because
``atexit`` runs *before* the interpreter's final garbage collection, calling
``Kokkos::finalize()`` there naively races the teardown of any NeoN ``Vector`` /
``MeshAdapter`` still alive — most often one pinned by a failed solve's traceback
in ``sys.last_*``. Its destructor then deallocates after ``Kokkos::finalize`` and
Kokkos calls ``host_abort`` (SIGABRT), which also clobbers the real init error in
the log. This pins the ordering fix: a process that finishes with a NeoN object
pinned by ``sys.last_traceback`` must still exit cleanly, no ``host_abort``.

The abort only fires at interpreter shutdown, so it cannot be observed in-process
— the test drives a subprocess and asserts on its exit code and stderr.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("neon._neon")  # skip when the NeoN bindings are not built

import neon._neon as nn  # noqa: E402

from neofoam.solver import neon_runtime  # noqa: E402
from neofoam.solver.neon_runtime import (  # noqa: E402
    _gpu_init_report,
    ensure_neon_initialized,
    requested_executor,
)
from neofoam.tooling.casebuild import from_template, patch  # noqa: E402

#: The lid-driven cavity case; its controlDict carries ``executor Serial``.
SETUP_PIMPLE = Path(__file__).parents[1] / "setup_pimple"

# An ordinal no machine exposes: Kokkos selects it, the CUDA runtime rejects it,
# and the healthy device is never touched.
_ABSENT_DEVICE_ORDINAL = "4095"

# os._exit after printing: Kokkos' own static teardown segfaults once its GPU
# backend failed to come up (pre-existing, and unrelated to the message this
# test is about), which would otherwise hide whether the error was catchable.
_FORCED_GPU_INIT_FAILURE = """
import os
import sys

from neofoam.solver.neon_runtime import ensure_neon_initialized

try:
    ensure_neon_initialized(["forced-gpu-init-failure"])
except RuntimeError as error:
    print(error)
    sys.stdout.flush()
    os._exit(0)

print("NO GPU BACKEND IN THIS BUILD")
sys.stdout.flush()
os._exit(0)
"""

# A failed solve, reduced to essentials: initialize through the shared guard,
# build a Kokkos-backed NeoN Vector as a local, raise, and record the exception
# in sys.last_* exactly as a top-level traceback print does. The Vector is now
# reachable only through the stored traceback and survives to final GC — the
# precise condition that aborted the process before the finalize ordering fix.
_PINNED_BY_TRACEBACK = """
import sys
from neofoam.solver.neon_runtime import ensure_neon_initialized
import neon._neon as nn

ensure_neon_initialized(["regression"])

def failing_solve():
    field = nn.ScalarVector(nn.SerialExecutor(), 128, 1.0)
    raise RuntimeError("InitStep failed holding " + repr(field))

try:
    failing_solve()
except RuntimeError:
    sys.last_type, sys.last_value, sys.last_traceback = sys.exc_info()

print("solve failed and recorded")
"""


def test_requested_executor_reads_the_controlDict_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ``executor`` entry of system/controlDict names the executor."""
    case = (from_template(SETUP_PIMPLE) | patch("system/controlDict", executor="GPU")).build_at(
        tmp_path / "gpuExecutor"
    )
    monkeypatch.chdir(case.path)

    assert requested_executor() == "GPU"


def test_requested_executor_without_the_entry_defaults_to_serial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stock case carrying no NeoN keys still runs, on the host executor."""
    case = (
        from_template(SETUP_PIMPLE) | patch("system/controlDict", remove=["executor"])
    ).build_at(tmp_path / "noExecutorEntry")
    monkeypatch.chdir(case.path)

    assert requested_executor() == "Serial"


def test_gpu_init_report_names_the_error_the_executor_and_the_visibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The report restates the Kokkos error with the state that explains it."""
    monkeypatch.chdir(SETUP_PIMPLE)
    monkeypatch.setenv("KOKKOS_VISIBLE_DEVICES", "4095")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    report = _gpu_init_report(RuntimeError("cudaErrorInvalidDevice"))

    assert "NeoN (Kokkos) initialization failed on the GPU backend" in report
    assert "cudaErrorInvalidDevice" in report
    assert "requested executor: Serial (system/controlDict)" in report
    assert "CUDA_VISIBLE_DEVICES=<unset>" in report
    assert "KOKKOS_VISIBLE_DEVICES=4095" in report
    assert "Kokkos::initialize brings up every compiled backend" in report
    assert "doc/reference/cli.rst" in report


def test_non_cuda_init_failure_is_reraised_unwrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only a CUDA failure gets the GPU report; wrapping all would bury other causes."""
    monkeypatch.setattr(neon_runtime, "_neon_initialized", False)
    original = RuntimeError("Kokkos::initialize: unrecognised command line argument")

    def failing_initialize(argv: list[str]) -> None:
        raise original

    monkeypatch.setattr(nn, "initialize", failing_initialize)

    with pytest.raises(RuntimeError) as raised:
        ensure_neon_initialized(["unit-test"])

    assert raised.value is original


def test_failed_gpu_init_reports_the_executor_and_the_ways_out() -> None:
    """A real GPU init failure is catchable and carries the diagnostic."""
    env = {**os.environ, "KOKKOS_VISIBLE_DEVICES": _ABSENT_DEVICE_ORDINAL}
    result = subprocess.run(
        [sys.executable, "-c", _FORCED_GPU_INIT_FAILURE],
        cwd=str(SETUP_PIMPLE),
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )

    if "NO GPU BACKEND IN THIS BUILD" in result.stdout:
        pytest.skip("host-only Kokkos build — no GPU backend init to fail")
    assert result.returncode == 0, (
        f"worker did not raise a catchable error (returncode={result.returncode})\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    report = result.stdout
    assert "NeoN (Kokkos) initialization failed on the GPU backend" in report
    # The underlying Kokkos message is kept, not swallowed.
    assert "cuda" in report.lower()
    assert "requested executor: Serial (system/controlDict)" in report
    assert f"KOKKOS_VISIBLE_DEVICES={_ABSENT_DEVICE_ORDINAL}" in report
    assert "doc/reference/cli.rst" in report


def test_finalize_does_not_abort_with_neon_object_pinned_by_traceback() -> None:
    """A NeoN Vector pinned by sys.last_traceback still finalizes cleanly."""
    result = subprocess.run(
        [sys.executable, "-c", _PINNED_BY_TRACEBACK],
        capture_output=True,
        text=True,
        timeout=120,
    )

    # -6 / 134 is SIGABRT from Kokkos host_abort; a clean run exits 0.
    assert result.returncode == 0, (
        f"neon teardown aborted (returncode={result.returncode})\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "deallocated after Kokkos::finalize" not in result.stderr
    assert "solve failed and recorded" in result.stdout
