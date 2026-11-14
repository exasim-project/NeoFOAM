# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Advection-only VoF: transport ``alpha`` with MULES in a prescribed velocity."""

from __future__ import annotations

import math
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Protocol

import numpy as np
import pybFoam as pyf

from neofoam.solver.incompressibleVoF.configs import ControlDictConfig

from ._neon_backend import NeoNBackend
from ._openfoam_backend import OpenFOAMBackend

Backend = Literal["openfoam", "neon"]

VelocityFunction = Callable[[np.ndarray, float], np.ndarray]
"""A prescribed velocity ``U(x, t)``: cell centres ``(n, 3)`` and time to ``(n, 3)``."""

# OpenFOAM's SMALL, which setDeltaT.H adds to the Courant numbers.
_SMALL = 1e-15


@dataclass(frozen=True)
class AdvectionResult:
    """``alpha`` at the start time and at every write time of one run.

    ``alpha[i]`` is the internal field at ``times[i]``; ``cell_volumes`` are the
    mesh cell volumes, so ``alpha @ cell_volumes`` is the phase volume per write
    time. Plain numpy, so it crosses process boundaries (see
    :func:`advect_in_subprocess`).

    Example::

        result = advect(case, "openfoam")
        volume_drift = result.alpha @ result.cell_volumes - result.alpha[0] @ result.cell_volumes
    """

    times: np.ndarray
    alpha: np.ndarray
    cell_volumes: np.ndarray


class _Backend(Protocol):
    """One time step of alpha transport, on the backend's own fields."""

    def alpha(self) -> np.ndarray: ...
    def cell_volumes(self) -> np.ndarray: ...
    def courant_numbers(self, delta_t: float) -> tuple[float, float]: ...
    def advance(self) -> None: ...
    def write(self) -> None: ...
    def scale_flux(self, factor: float) -> None: ...


@dataclass
class _FluxModulation:
    """advectorVoF's time dependence of the initial flux (its ``isoAdvector`` dict)."""

    period: float
    spiralling_flow: float
    reverse_time: float
    sign: float = 1.0

    @classmethod
    def read(cls) -> Optional[_FluxModulation]:
        """The case's modulation, or ``None`` when the flux is constant in time."""
        fv_solution = pyf.dictionary.read("system/fvSolution")
        if not fv_solution.found("isoAdvector"):
            return None
        controls = fv_solution.subDict("isoAdvector")
        modulation = cls(
            period=float(controls.getOrDefault[float]("period", 0.0)),
            spiralling_flow=float(controls.getOrDefault[float]("spirallingFlow", 0.0)),
            reverse_time=float(controls.getOrDefault[float]("reverseTime", 0.0)),
        )
        if max(modulation.period, modulation.spiralling_flow, modulation.reverse_time) <= 0:
            return None
        return modulation

    def factor(self, time: float, delta_t: float) -> float:
        """``phi/phi0`` for the step from ``time`` of length ``delta_t``."""
        if self.reverse_time > 0 and time >= self.reverse_time:
            self.sign = -self.sign
            self.reverse_time = -1.0
        # advectorVoF's operation order, so the cosine is the same double; negating
        # phi0 or the factor is exact, so phi is bitwise advectorVoF's.
        factor = self.sign
        if self.period > 0:
            factor = self.sign * math.cos(2.0 * math.pi * (time + 0.5 * delta_t) / self.period)
        if self.spiralling_flow > 0:
            factor = self.sign * math.cos(math.pi * (time + 0.5 * delta_t) / self.spiralling_flow)
        return factor


def advect(
    case: Path,
    backend: Backend = "openfoam",
    executor: str = "Serial",
    write: bool = False,
    velocity: VelocityFunction | None = None,
) -> AdvectionResult:
    """Advect the case's ``alpha.<phase>`` from ``startTime`` to ``endTime``.

    Use it from a process that runs one case; to run several (backends,
    executors, cases) from one Python program use :func:`advect_in_subprocess`.
    ``case`` must be meshed and hold ``alpha.<phase>`` and ``U`` at the start
    time. ``executor`` only matters for the neon backend. ``write`` also writes
    ``alpha.<phase>`` into the case's time directories at the write times.
    ``velocity`` replaces ``U`` by ``U(cell centres, t)``, evaluated at the
    middle of every step (openfoam backend only); without it ``U`` is read once
    and held fixed. The working directory is the case for the duration of the
    call. Function objects are disabled.

    The equivalent of TwoPhaseFlow's ``advectorVoF`` with its MULES scheme: no
    momentum and no pressure, only the phase fraction moving through a fixed flux
    ``phi = fvc::flux(U)``. Two interchangeable backends run the same time loop:

    * ``"openfoam"`` — OpenFOAM's own MULES through pybFoam.
    * ``"neon"`` — NeoFOAM's NeoN MULES kernels on the ``Serial``, ``CPU`` or
      ``GPU`` executor.

    Both run the ``alphaEqn.H`` loop of :mod:`neofoam.algorithms.mules`, which the
    ``incompressibleVoF`` solver uses too.

    Both read the same case inputs: the mesh, ``<startTime>/alpha.<phase>`` and
    ``<startTime>/U``; the MULES controls of the ``alpha.<phase>`` solver dict in
    ``system/fvSolution`` (``nAlphaCorr``, ``nAlphaSubCycles``, ``cAlpha``,
    ``MULESCorr``, ``nLimiterIter``, ``alphaApplyPrevCorr``); ``div(phi,alpha)``
    and ``div(phirb,alpha)`` from ``system/fvSchemes``; and the time controls of
    ``system/controlDict``. With ``adjustTimeStep yes`` the step follows
    advectorVoF's ``setDeltaT.H``: undamped, called once before the loop and then
    every step, ``deltaT *= min(maxCo/Co, maxAlphaCo/alphaCo)`` capped by
    ``maxDeltaT``, where ``alphaCo`` only counts cells with
    ``0.01 < alpha < 0.99``. The MULES source is interFoam's ``Su = 0``, not
    advectorVoF's ``Su = div(phi)*min(alpha, 1)``: the two agree when ``phi`` is
    divergence-free.

    The flux is ``<startTime>/phi`` when the case has one (``createPhi.H``),
    otherwise ``fvc::flux(U)``. An ``isoAdvector`` dict in ``system/fvSolution``
    makes it time-dependent as in advectorVoF: every step, after ``setDeltaT``
    and before the time is advanced, ``reverseTime`` (once ``t >= reverseTime``)
    negates the initial flux ``phi0``, then ``period`` sets
    ``phi = phi0*cos(2 pi (t + dt/2)/period)`` and ``spirallingFlow``
    ``phi = phi0*cos(pi (t + dt/2)/spirallingFlow)``; ``t`` is the time at the
    start of the step and ``dt`` the step just set. The Courant numbers of the
    next ``setDeltaT`` see that flux. Both backends apply it the same way.

    Supported MULES paths: both backends run every ``alphaEqn.H`` branch
    (``MULESCorr`` yes/no, any ``nAlphaCorr``, sub-cycling, ``alphaApplyPrevCorr``,
    Euler/CrankNicolson ``ddt(alpha)``). The neon backend solves the
    ``MULESCorr`` implicit predictor with Ginkgo, mapped from the alpha solver
    dict, so it agrees with the openfoam backend to the solver tolerance rather
    than to round-off; it implements only ``Gauss vanLeer`` / ``Gauss linear``
    and a ``Gauss linear`` gradient, and raises ``NotImplementedError`` for other
    schemes.

    Process model: a run constructs the process's one ``Foam::Time``, and the neon
    backend initializes Kokkos, which can happen once per process. So a process can
    call :func:`advect` once. To compare backends or executors, run each in its own
    process with :func:`advect_in_subprocess`.

    Example::

        result = advect(Path("discUniFlow"), "neon", "GPU")
        print(result.times, result.alpha.shape)
    """
    with _working_directory(case):
        # Foam::Time keeps a reference to the argList: it must outlive the run.
        arg_list = pyf.argList(["advect", "-noFunctionObjects"])
        runtime = pyf.Time(arg_list)
        transport = _create_backend(runtime, backend, executor, velocity)
        return _run(runtime, transport, write)


def advect_in_subprocess(
    case: Path,
    backend: Backend = "openfoam",
    executor: str = "Serial",
    write: bool = False,
) -> AdvectionResult:
    """:func:`advect` in a fresh Python process, so one program can run many cases.

    Each call gets its own ``Foam::Time`` and Kokkos runtime, which is what
    comparing backends or executors needs. The solver log goes to
    ``<case>/log.advect.<backend>.<executor>``. A prescribed velocity function
    cannot cross the process boundary; call :func:`advect` in your own worker
    for that.

    Example::

        reference = advect_in_subprocess(case_a, "openfoam")
        subject = advect_in_subprocess(case_b, "neon", "GPU")
    """
    case = case.resolve()
    log_path = case / f"log.advect.{backend}.{executor}"
    with tempfile.TemporaryDirectory() as scratch:
        out = Path(scratch) / "result.npz"
        arguments = [str(case), backend, executor, "1" if write else "0", str(out)]
        with log_path.open("w") as log:
            completed = subprocess.run(
                [sys.executable, "-m", "neofoam.algorithms.vof._worker", *arguments],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if completed.returncode != 0:
            raise RuntimeError(
                f"advect({case}, {backend!r}, {executor!r}) failed with exit code "
                f"{completed.returncode}; log tail:\n{log_path.read_text()[-4000:]}"
            )
        with np.load(out) as saved:
            return AdvectionResult(saved["times"], saved["alpha"], saved["cell_volumes"])


@contextmanager
def _working_directory(case: Path) -> Iterator[None]:
    """Run inside ``case``: pybFoam and the dictionary readers resolve paths from there."""
    previous = Path.cwd()
    os.chdir(case)
    try:
        yield
    finally:
        os.chdir(previous)


def _create_backend(
    runtime: pyf.Time,
    backend: Backend,
    executor: str,
    velocity: VelocityFunction | None,
) -> _Backend:
    """The backend's fields and stepper, on the already constructed ``runtime``."""
    alpha_name = _alpha_field_name(Path(str(runtime.timeName())))
    if backend == "openfoam":
        return OpenFOAMBackend(runtime, alpha_name, velocity)
    if backend == "neon":
        if velocity is not None:
            raise NotImplementedError(
                "a prescribed velocity function needs the flux on the NeoN executor "
                "refreshed every step, which the NeoFOAM bindings cannot do yet "
                "(no host-to-device copy for a surface field); use backend='openfoam'"
            )
        return NeoNBackend(runtime, alpha_name, executor)
    raise ValueError(f"unknown backend {backend!r}: expected 'openfoam' or 'neon'")


def _alpha_field_name(start_time_dir: Path) -> str:
    """The one ``alpha.<phase>`` field of the start time directory."""
    names = sorted(
        path.name for path in start_time_dir.glob("alpha.*") if not path.name.endswith(".orig")
    )
    if len(names) != 1:
        raise ValueError(
            f"expected exactly one alpha.<phase> field in {start_time_dir.resolve()}, found {names}"
        )
    return names[0]


def _run(runtime: pyf.Time, transport: _Backend, write: bool) -> AdvectionResult:
    """advectorVoF's time loop around the backend's alpha step."""
    controls = ControlDictConfig.load(validate=False)
    modulation = _FluxModulation.read()
    volumes = transport.cell_volumes()
    times = [runtime.value()]
    alphas = [transport.alpha()]

    _set_delta_t(runtime, transport, controls)
    while runtime.run():
        _set_delta_t(runtime, transport, controls)
        if modulation is not None:
            transport.scale_flux(modulation.factor(runtime.value(), runtime.deltaTValue()))
        runtime.increment()
        transport.advance()
        if runtime.outputTime():
            times.append(runtime.value())
            alphas.append(transport.alpha())
            if write:
                transport.write()

    return AdvectionResult(np.array(times), np.array(alphas), volumes)


def _set_delta_t(runtime: pyf.Time, transport: _Backend, controls: ControlDictConfig) -> None:
    """advectorVoF's ``setDeltaT.H``: undamped, limited by both Courant numbers."""
    if not controls.adjustTimeStep:
        return
    co_num, alpha_co_num = transport.courant_numbers(runtime.deltaTValue())
    factor = min(
        controls.maxCo / (co_num + _SMALL),
        controls.maxAlphaCo / (alpha_co_num + _SMALL),
    )
    runtime.setDeltaT(min(factor * runtime.deltaTValue(), controls.maxDeltaT))
