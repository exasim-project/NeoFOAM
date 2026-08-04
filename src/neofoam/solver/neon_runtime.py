# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Process-wide NeoN (Kokkos) runtime initialization.

Kokkos may only be initialized/finalized once per process. Every NeoN-backed
solver calls :func:`ensure_neon_initialized` before touching NeoN so repeated
solver runs in one process (e.g. the test suite) reuse a single initialization;
finalization is registered once at interpreter exit.

The executor the NeoN fields live on comes from :func:`requested_executor`.
It selects where the data is placed, *not* which Kokkos backends come up:
``Kokkos::initialize`` brings up every backend the library was compiled with,
so a CUDA-enabled build takes a CUDA context per rank even for a Serial run.
When that context cannot be created, :func:`ensure_neon_initialized` re-raises
with the device inventory and the ways out.
"""

import atexit
import gc
import os
import subprocess
import sys

import neon._neon as nn  # NeoN Python bindings

_neon_initialized = False

_EXECUTOR_ENV = "NEOFOAM_EXECUTOR"

# The launcher-specific rank variables, most specific first; only one is set.
_RANK_ENVS = ("OMPI_COMM_WORLD_RANK", "PMIX_RANK", "PMI_RANK", "SLURM_PROCID")


def requested_executor() -> str:
    """NeoN executor the solver places its fields on.

    Read once per process from ``NEOFOAM_EXECUTOR`` (``Serial``, ``CPU``,
    ``GPU`` or ``default``), defaulting to the deterministic host executor
    ``Serial`` that the test suite and the verification sweep run on. Use the
    ``-executor`` argument instead when driving the C++ solvers directly.

    Example::

        NEOFOAM_EXECUTOR=GPU neofoam solver incompressiblefluidneon
    """
    return os.environ.get(_EXECUTOR_ENV, "Serial")


def _launcher_rank() -> str:
    """The MPI rank of this process as its launcher reports it."""
    for name in _RANK_ENVS:
        if name in os.environ:
            return f"{os.environ[name]} (${name})"
    return "unknown — not launched by mpirun/srun"


def _device_inventory() -> str:
    """One indented line per GPU: index, name, free and total memory."""
    query = "--query-gpu=index,name,memory.free,memory.total"
    try:
        listing = subprocess.run(
            ["nvidia-smi", query, "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError) as probe_error:
        return f"  nvidia-smi did not report: {probe_error}"
    lines = listing.split("\n")
    return "\n".join(f"  {line}" for line in lines if line.strip()) or "  no device reported"


def _gpu_init_report(error: BaseException) -> str:
    """The failed Kokkos GPU init, restated with the device state and the ways out."""
    seen_by_kokkos = "\n".join(
        f"{name}={os.environ.get(name, '<unset>')}"
        for name in ("CUDA_VISIBLE_DEVICES", "KOKKOS_VISIBLE_DEVICES")
    )
    return (
        f"NeoN (Kokkos) initialization failed on the GPU backend: {error}\n"
        f"requested executor: {requested_executor()} (${_EXECUTOR_ENV})\n"
        f"MPI rank: {_launcher_rank()}\n"
        f"{seen_by_kokkos}\n"
        f"devices:\n{_device_inventory()}\n"
        "This NeoFOAM links a Kokkos compiled with the CUDA backend, and "
        "Kokkos::initialize brings up every compiled backend — so each rank takes a "
        "CUDA context (a few hundred MiB) even when the executor is Serial or CPU. "
        f"${_EXECUTOR_ENV} cannot switch that off. Ways out:\n"
        "  * free the device, or select another one: KOKKOS_VISIBLE_DEVICES=<index above>\n"
        "  * run fewer ranks per device — every rank takes its own context\n"
        "  * for a host-only run, use a NeoFOAM built with -DKokkos_ENABLE_CUDA=OFF\n"
    )


def _finalize_neon() -> None:
    """Finalize Kokkos, but only after Python-owned NeoN objects are gone.

    ``atexit`` handlers run *before* the interpreter's final garbage collection,
    so calling ``nn.finalize()`` (``Kokkos::finalize()``) directly here races the
    teardown of any NeoN ``Vector`` / ``MeshAdapter`` still reachable at exit.
    The usual culprit is a failed solve: the exception it raised is kept in
    ``sys.last_*`` (and, on 3.11+, ``sys.last_exc``), whose traceback pins the
    frames — and thus the partially-built runtime and fields — alive. Those
    destructors then deallocate *after* ``Kokkos::finalize`` and Kokkos aborts the
    process (``host_abort``), which also clobbers the real init error in the log.

    Dropping the stored exception and forcing a collection destroys those objects
    first, so ``finalize`` (and its ginkgo executor-release hook) runs cleanly.
    """
    for _attr in ("last_traceback", "last_value", "last_type", "last_exc"):
        if hasattr(sys, _attr):
            setattr(sys, _attr, None)
    gc.collect()
    nn.finalize()


def ensure_neon_initialized(argv: list[str]) -> None:
    """Initialize NeoN/Kokkos once per process (idempotent)."""
    global _neon_initialized
    if _neon_initialized:
        return
    try:
        nn.initialize(argv)
    except RuntimeError as error:
        # Kokkos reports a bare CUDA error code from a frame no user can act on;
        # anything else is passed through untouched.
        if "cuda" not in str(error).lower():
            raise
        raise RuntimeError(_gpu_init_report(error)) from error
    _neon_initialized = True
    atexit.register(_finalize_neon)
