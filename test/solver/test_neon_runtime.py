# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Process-wide NeoN/Kokkos initialization: executor selection and teardown.

``requested_executor``, ``_launcher_rank`` and the report wording are environment
reads and string formatting, tested in-process — so the wording holds on a
host-only build too, where the subprocess test below has nothing to provoke and
skips. The other two behaviors are only observable across a whole process life,
hence subprocess-driven:

**A failed GPU init must say what to do about it.** Kokkos brings up every
backend it was compiled with, so a CUDA-enabled build takes a CUDA context per
rank even for a Serial executor; when that fails Kokkos reports a bare CUDA
error code and a Kokkos source line, which names neither the device, the rank,
nor a remedy. ``KOKKOS_VISIBLE_DEVICES`` pointing at a device ordinal that
cannot exist forces exactly that failure without disturbing the real device, so
the diagnostic can be asserted on a machine whose GPU is healthy. A build
without a GPU backend initializes fine and has no diagnostic to show — that run
is reported by the worker and skipped.

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

import pytest

pytest.importorskip("neon._neon")  # skip when the NeoN bindings are not built

import neon._neon as nn  # noqa: E402

from neofoam.solver import neon_runtime  # noqa: E402
from neofoam.solver.neon_runtime import (  # noqa: E402
    _RANK_ENVS,
    _gpu_init_report,
    _launcher_rank,
    ensure_neon_initialized,
    requested_executor,
)

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


def test_requested_executor_defaults_to_serial(monkeypatch: pytest.MonkeyPatch) -> None:
    """With NEOFOAM_EXECUTOR unset the host executor Serial is selected."""
    monkeypatch.delenv("NEOFOAM_EXECUTOR", raising=False)

    assert requested_executor() == "Serial"


def test_requested_executor_reads_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """NEOFOAM_EXECUTOR names the executor."""
    monkeypatch.setenv("NEOFOAM_EXECUTOR", "GPU")

    assert requested_executor() == "GPU"


@pytest.mark.parametrize("rank_env", _RANK_ENVS)
def test_launcher_rank_names_the_variable_it_read(
    monkeypatch: pytest.MonkeyPatch, rank_env: str
) -> None:
    """Every launcher's rank variable is recognised, and the report says which."""
    for name in _RANK_ENVS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(rank_env, "3")

    assert _launcher_rank() == f"3 (${rank_env})"


def test_launcher_rank_prefers_the_most_specific_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """srun inside an OpenMPI job sets both; the OpenMPI rank is the process's own."""
    for name in _RANK_ENVS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SLURM_PROCID", "0")
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "7")

    assert _launcher_rank() == "7 ($OMPI_COMM_WORLD_RANK)"


def test_launcher_rank_without_a_launcher_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    """A plain (non-MPI) run reports no rank rather than a misleading 0."""
    for name in _RANK_ENVS:
        monkeypatch.delenv(name, raising=False)

    assert _launcher_rank() == "unknown — not launched by mpirun/srun"


def test_gpu_init_report_names_the_devices_and_the_ways_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The report restates the Kokkos error with the state that explains it.

    ``_device_inventory`` shells out to ``nvidia-smi``, hence the stub.
    """
    monkeypatch.setenv("NEOFOAM_EXECUTOR", "Serial")
    monkeypatch.setenv("KOKKOS_VISIBLE_DEVICES", "4095")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    for name in _RANK_ENVS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(neon_runtime, "_device_inventory", lambda: "  0, NVIDIA A100, 1 MiB, 2 MiB")

    report = _gpu_init_report(RuntimeError("cudaErrorInvalidDevice"))

    assert "NeoN (Kokkos) initialization failed on the GPU backend" in report
    assert "cudaErrorInvalidDevice" in report
    assert "requested executor: Serial ($NEOFOAM_EXECUTOR)" in report
    assert "MPI rank: unknown — not launched by mpirun/srun" in report
    assert "CUDA_VISIBLE_DEVICES=<unset>" in report
    assert "KOKKOS_VISIBLE_DEVICES=4095" in report
    assert "devices:\n  0, NVIDIA A100, 1 MiB, 2 MiB" in report
    assert "KOKKOS_VISIBLE_DEVICES=<index above>" in report
    assert "-DKokkos_ENABLE_CUDA=OFF" in report


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


def test_failed_gpu_init_reports_the_devices_and_the_ways_out() -> None:
    """A GPU init failure names the executor, the rank, the devices and the remedies."""
    env = {
        **os.environ,
        "KOKKOS_VISIBLE_DEVICES": _ABSENT_DEVICE_ORDINAL,
        "NEOFOAM_EXECUTOR": "Serial",
    }
    result = subprocess.run(
        [sys.executable, "-c", _FORCED_GPU_INIT_FAILURE],
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
    assert "requested executor: Serial ($NEOFOAM_EXECUTOR)" in report
    assert "MPI rank:" in report
    assert f"KOKKOS_VISIBLE_DEVICES={_ABSENT_DEVICE_ORDINAL}" in report
    assert "devices:" in report
    assert "KOKKOS_VISIBLE_DEVICES=<index above>" in report
    assert "-DKokkos_ENABLE_CUDA=OFF" in report


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
