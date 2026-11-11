# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Process-wide NeoN (Kokkos) runtime initialization, and a NeoN run's mesh geometry and fields.

Kokkos may only be initialized/finalized once per process. Every NeoN-backed
solver calls :func:`ensure_neon_initialized` before touching NeoN so repeated
solver runs in one process (e.g. the test suite) reuse a single initialization;
finalization is registered once at interpreter exit.

The executor the NeoN fields live on comes from :func:`requested_executor`,
which selects where the data is placed, *not* which Kokkos backends come up:
``Kokkos::initialize`` brings up every backend the library was compiled with,
so a CUDA-enabled build takes a CUDA context per rank even for a Serial run.
When that context cannot be created, :func:`ensure_neon_initialized` re-raises
with the state that explains it; the remedies live in ``doc/reference/cli.rst``.
"""

import atexit
import gc
import os
import sys
from typing import Any, Mapping

import neon._neon as nn  # NeoN Python bindings
import numpy as np

from neofoam import neofoam_bindings as nfb
from neofoam.framework.coupling import Patch
from neofoam.io import OF, BaseConfig, IOStrategy

_neon_initialized = False


@IOStrategy(OF("system/controlDict"))
class NeoNControlConfig(BaseConfig):
    """The NeoN runtime keys of ``system/controlDict`` — today the executor alone.

    A top-level (sub-dict-free) slice of the case's ``controlDict``, co-owning the
    file with the solver's ``ControlDictConfig`` the way the ``courant`` /
    ``maxDeltaT`` model configs do. It is deliberately *not* a field on
    ``ControlDictConfig``: it is read by the shared NeoN runtime (the framework
    solver and the legacy ``neoPimpleFoam`` alike), and modelling one key keeps
    that read independent of the time-control entries. ``executor`` stays a plain
    ``str`` rather than a ``Literal``: the names are resolved by NeoFOAM's C++
    ``createExecutor``, and an unknown one must fail there — after it logged
    ``Creating Executor <name>`` — not at config validation.

    Declared on the ``incompressibleFluidNeoN`` spec, so it shows up in the
    solver's config schema; read through :func:`requested_executor`.

    Example::

        executor        GPU;   // in system/controlDict
    """

    executor: str = "Serial"


def requested_executor() -> str:
    """NeoN executor the solver places its fields on.

    Read from the ``executor`` entry of ``system/controlDict`` (``Serial``,
    ``CPU``, ``GPU`` or ``default``) via :class:`NeoNControlConfig` — the same
    entry the C++ solvers read. A case without the entry (a stock pimpleFoam
    case) falls back to the deterministic host executor ``Serial``. The case is
    the working directory, as for every other case file the solvers read. Use
    the ``-executor`` argument instead when driving the C++ solvers directly.

    Example::

        executor        GPU;   // in system/controlDict
    """
    return NeoNControlConfig.load().executor


def _gpu_init_report(error: BaseException) -> str:
    """The failed Kokkos GPU init, restated with the state that explains it."""
    seen_by_kokkos = "\n".join(
        f"{name}={os.environ.get(name, '<unset>')}"
        for name in ("CUDA_VISIBLE_DEVICES", "KOKKOS_VISIBLE_DEVICES")
    )
    return (
        f"NeoN (Kokkos) initialization failed on the GPU backend: {error}\n"
        f"requested executor: {requested_executor()} (system/controlDict)\n"
        f"{seen_by_kokkos}\n"
        "This NeoFOAM links a Kokkos compiled with the CUDA backend, and "
        "Kokkos::initialize brings up every compiled backend — so each rank takes a "
        "CUDA context (a few hundred MiB) even when the executor is Serial or CPU. "
        "The controlDict executor entry cannot switch that off. Ways out: see "
        "doc/reference/cli.rst, 'Executor selection and GPU initialization'.\n"
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


def _host(vector: Any) -> Any:
    """A NeoN vector as a numpy array on the host."""
    return np.asarray(vector.copy_to_host())


class NeoNGeometry:
    """The :class:`~neofoam.framework.coupling.MeshGeometry` of a NeoN run, from its mesh.

    A patch's faces are its slice of NeoN's boundary faces, flat in patch order.

    Example::

        ctx.models["coupling"] = Coupling(coupling, NeoNGeometry(rt), NeoNFields(rt, ctx.fields))
    """

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    def faces(self, patch: str) -> slice:
        """The slice of NeoN's boundary faces that are ``patch``'s."""
        index = int(self.runtime.mesh.boundary().findPatchID(patch))
        offset = self.runtime.nf_mesh.boundary_mesh().offset()
        return slice(offset[index], offset[index + 1])

    def face_centres(self, patch: str) -> Any:
        return _host(self.runtime.nf_mesh.boundary_mesh().face_centers())[self.faces(patch)]

    def face_areas(self, patch: str) -> Any:
        return _host(self.runtime.nf_mesh.boundary_mesh().face_areas())[self.faces(patch)]

    def cell_centres(self) -> Any:
        return _host(self.runtime.nf_mesh.cell_centers)

    def cell_volumes(self) -> Any:
        return _host(self.runtime.nf_mesh.cell_volumes)


class NeoNFields:
    """The :class:`~neofoam.framework.coupling.FieldAccess` of a NeoN run's volume fields.

    A patch reads its boundary values and is written as fixed values; the cells
    read and write the internal values. A patch it writes is ``calculated`` in
    ``0/<field>``: a boundary correction re-applies any other condition.
    """

    def __init__(self, runtime: Any, fields: Mapping[str, Any]) -> None:
        self.geometry = NeoNGeometry(runtime)
        self.fields = fields

    def read(self, field: str, location: Any) -> Any:
        if isinstance(location, Patch):
            values = self.fields[field].boundary_data_value()
            return _host(values)[self.geometry.faces(location.name)]
        return _host(self.fields[field].internal_vector())

    def write(self, field: str, location: Any, values: Any) -> None:
        values = np.ascontiguousarray(values, dtype=float)
        if isinstance(location, Patch):
            index = int(self.geometry.runtime.mesh.boundary().findPatchID(location.name))
            self.fields[field].set_patch_fixed_values(index, values)
        else:
            nfb.copy_from_host(self.fields[field], values)
