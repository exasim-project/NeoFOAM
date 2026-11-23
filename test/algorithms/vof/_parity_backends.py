# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The ``pyb`` / ``nb`` VoF backends and the case handles the parity tests call them on.

Both backends hold per-process global state (one ``Foam::Time``, one Kokkos
runtime), so every call is forwarded to a persistent worker subprocess bound
to the staged case of the field's mesh and executor. The wire protocol is JSON
lines on stdin/stdout: requests carry an optional ``data`` .npy path and
answers land as ``<out>.npy`` files; the ``#RESULT`` reply prefix separates
them from OpenFOAM/NeoN log noise.

Fields read and write as numpy: ``np.asarray(field)`` fetches the internal
field, ``field[:] = values`` writes it on **both** backends (the pybFoam worker
persists ``0/<name>`` with 17 digits, the neon worker reloads it).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

_RESULT_PREFIX = "#RESULT "


def _assert_relative_l2(result: np.ndarray, expected: np.ndarray, tol: float, msg: str) -> None:
    """Assert ``||result - expected||_2 / ||expected||_2 <= tol`` over all entries."""
    assert result.shape == expected.shape, f"{msg}: shape {result.shape} vs {expected.shape}"
    error = np.linalg.norm(result - expected) / np.linalg.norm(expected)
    assert error <= tol, f"{msg}: relative L2 error {error:.3e} > {tol:.0e}"


class _Worker:
    """One persistent evaluation subprocess bound to a staged case."""

    def __init__(self, script: Path, case_dir: Path, label: str, argv: list[str]):
        self._out_dir = case_dir / "_results" / label
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self._out_dir / "worker.log"
        self._log = self._log_path.open("w")
        env = os.environ.copy()
        env["FOAM_SIGFPE"] = "false"
        self._proc = subprocess.Popen(
            [sys.executable, str(script), str(case_dir), *argv],
            cwd=case_dir,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log,
            text=True,
        )
        self._count = 0

    def wait_ready(self) -> None:
        """Block until the worker prints ``#READY`` (case staged, ops built)."""
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            if line.startswith("#READY"):
                return
        raise RuntimeError(
            f"worker exited before becoming ready "
            f"(exit code {self._proc.poll()}, log: {self._log_path})"
        )

    def _request(self, func: str, args: tuple[str, ...], data: np.ndarray | None) -> Path:
        self._count += 1
        out = self._out_dir / f"{self._count:04d}.npy"
        request = {"func": func, "args": list(args), "out": str(out)}
        if data is not None:
            data_path = self._out_dir / f"{self._count:04d}_in.npy"
            np.save(data_path, data)
            request["data"] = str(data_path)
        assert self._proc.stdin is not None and self._proc.stdout is not None
        self._proc.stdin.write(json.dumps(request) + "\n")
        self._proc.stdin.flush()
        for line in self._proc.stdout:
            if not line.startswith(_RESULT_PREFIX):
                continue
            reply = json.loads(line[len(_RESULT_PREFIX) :])
            if reply["status"] != "ok":
                raise RuntimeError(f"{func}{args} failed in worker: {reply['message']}")
            return out
        raise RuntimeError(
            f"worker exited while evaluating {func}{args} "
            f"(exit code {self._proc.poll()}, log: {self._log_path})"
        )

    def evaluate(
        self, func: str, args: tuple[str, ...], data: np.ndarray | None = None
    ) -> np.ndarray:
        """Run an op that produces an array result, optionally shipping ``data`` along."""
        return np.load(self._request(func, args, data))

    def command(self, func: str, args: tuple[str, ...], data: np.ndarray | None = None) -> None:
        """Run an op for its side effect, optionally shipping ``data`` along."""
        self._request(func, args, data)

    def close(self) -> None:
        if self._proc.poll() is None and self._proc.stdin is not None:
            try:
                self._proc.stdin.write('{"exit": true}\n')
                self._proc.stdin.flush()
                self._proc.wait(timeout=60)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                self._proc.kill()
        self._log.close()


class _Field:
    """One named field of a staged case, realized in both backend workers."""

    def __init__(self, name: str, pool: Any, case: str, executor: str):
        self.name = name
        self.pool = pool
        self.case = case
        self.executor = executor

    def __array__(self) -> np.ndarray:
        """The internal field values, fetched from the pybFoam worker."""
        return self.pool.of_worker(self.case).evaluate("field.get", (self.name,))

    def __setitem__(self, key: slice, values: np.ndarray) -> None:
        """Write values into the field on both backends (device copy on GPU)."""
        updated = np.asarray(self)
        updated[key] = values
        self.pool.of_worker(self.case).command("field.set", (self.name,), data=updated)
        self.pool.neon_worker(self.case, self.executor).command("field.reload", (self.name,))


@dataclass(frozen=True)
class _Mesh:
    """Mesh geometry as plain numpy arrays."""

    cell_centres: np.ndarray
    cell_volumes: np.ndarray


@dataclass(frozen=True)
class _Simulation:
    """One staged case bound to one executor."""

    pool: Any
    case: str
    executor: str
    mesh: _Mesh

    def field(self, name: str) -> _Field:
        if not (self.pool.case_dir(self.case) / "0" / name).is_file():
            raise ValueError(f"no 0/{name} in case {self.case}")
        return _Field(name, self.pool, self.case, self.executor)


def _use_velocity(sim: _Simulation, rotate: bool = True) -> _Field:
    """Copy the case's rotation or translation into ``U`` and rebuild ``phi`` on both backends."""
    velocity = sim.field("U")
    velocity[:] = np.asarray(sim.field("URotation" if rotate else "UTranslation"))
    sim.pool.of_worker(sim.case).command("flux.update", ("U",))
    sim.pool.neon_worker(sim.case, sim.executor).command("flux.update", ("U",))
    return _Field("phi", sim.pool, sim.case, sim.executor)


def _of_worker(field: _Field) -> _Worker:
    return field.pool.of_worker(field.case)  # type: ignore[no-any-return]


def _neon_worker(field: _Field) -> _Worker:
    return field.pool.neon_worker(field.case, field.executor)  # type: ignore[no-any-return]


class _VofOps:
    """The VoF face fluxes, on either backend's worker.

    ``alpha_phase_flux`` is interFoam's ``alphaPhiUn`` (vanLeer flux plus the
    ``c_alpha`` interface compression), ``compression_velocity`` and
    ``linear_compression_flux`` its compression pieces; all use the case's
    ``phi``; ``courant_numbers`` are the flow and interface Courant numbers.
    pybFoam composes them from ``fvc`` primitives, neon calls the NeoFOAM
    bindings.
    """

    def __init__(self, worker: Callable[[_Field], _Worker]) -> None:
        self._worker = worker

    def alpha_phase_flux(self, alpha: _Field, c_alpha: float, composed: bool = False) -> np.ndarray:
        """``composed``: neon builds it from the ``out=`` bindings (pybFoam ignores it)."""
        return self._worker(alpha).evaluate(
            "vof.alphaPhaseFlux", (alpha.name,), data=np.array([c_alpha, composed])
        )

    def compression_velocity(self, phi: _Field, c_alpha: float, into: bool = False) -> np.ndarray:
        """``phic`` of alphaEqn.H, internal then boundary face values; ``into``: neon's ``out=``."""
        return self._worker(phi).evaluate(
            "vof.compressionVelocity", (phi.name,), data=np.array([c_alpha, into])
        )

    def linear_compression_flux(self, alpha: _Field, into: bool = False) -> np.ndarray:
        """The linear compression flux of ``alpha`` with ``phir = phi``, internal then boundary."""
        return self._worker(alpha).evaluate(
            "vof.linearCompressionFlux", (alpha.name,), data=np.array([into])
        )

    def courant_numbers(self, alpha: _Field, delta_t: float) -> np.ndarray:
        """``[CoNum, alphaCoNum]`` of advectorVoF's setDeltaT.H for the case's ``phi``."""
        return self._worker(alpha).evaluate(
            "vof.courantNumbers", (alpha.name,), data=np.array([delta_t])
        )


class _MulesOps:
    """MULES on either backend's worker: ``pybFoam.mules`` vs the NeoFOAM bindings.

    Both return the advanced internal field and leave the case's field
    untouched. ``explicit_solve`` runs ``n_steps`` of the explicit alphaEqn
    with ``n_corr`` correctors (``alpha_phase_flux`` of the current alpha, then
    MULES advancing from the step's start); ``correct`` limits and applies
    ``alpha_phase_flux(alpha) - upwind_flux(base)``.
    """

    def __init__(self, worker: Callable[[_Field], _Worker]) -> None:
        self._worker = worker

    def explicit_solve(
        self, alpha: _Field, delta_t: float, c_alpha: float, n_steps: int, n_corr: int = 1
    ) -> np.ndarray:
        return self._worker(alpha).evaluate(
            "mules.explicitSolve",
            (alpha.name,),
            data=np.array([delta_t, c_alpha, n_steps, n_corr]),
        )

    def correct(self, alpha: _Field, base: _Field, delta_t: float, c_alpha: float) -> np.ndarray:
        return self._worker(alpha).evaluate(
            "mules.correct", (alpha.name, base.name), data=np.array([delta_t, c_alpha])
        )


class _Backend:
    """The VoF and MULES operators of one backend."""

    def __init__(self, worker: Callable[[_Field], _Worker]) -> None:
        self.vof = _VofOps(worker)
        self.mules = _MulesOps(worker)


pyb = _Backend(_of_worker)
nb = _Backend(_neon_worker)
