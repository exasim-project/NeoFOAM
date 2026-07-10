# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The ``pyb`` / ``nb`` backend singletons the tests call operators on.

Both backends hold per-process global state (one ``Foam::Time``, one Kokkos
runtime), so the test process cannot evaluate them directly on three meshes.
Every backend method forwards the call to a persistent worker subprocess
bound to the staged case of the field's ``case``/``executor`` (tests obtain
fields from ``case_setup.simulation(...)``), executes the real operator
there, and returns the internal field as a numpy array (neon results are
copied to host).

The mapping is one-to-one: ``pyb.fvc.div(phi, U, scheme=...)`` runs
``fvc.div(phi, U, scheme=...)`` in the pybFoam worker; ``nb.imp.div(...)``
assembles the implicit NeoN operator and applies its matrix. See
``of_worker.py`` / ``neon_worker.py`` for the exact per-operator code.

The wire protocol is JSON lines on stdin/stdout: requests carry an optional
``data`` .npy path (arrays pushed *into* the worker, e.g. field values set
by a test) and answers land as ``<out>.npy`` files; the ``#RESULT`` reply
prefix separates them from OpenFOAM/NeoN log noise. The pybFoam worker also
stages its case at startup (mesh generation via ``pybFoam.meshing``) and
signals ``#READY``; the ``WorkerPool`` in ``conftest.py`` awaits that before
starting neon workers.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from case_setup import Field

_RESULT_PREFIX = "#RESULT "

# The session's WorkerPool (staging + worker management); set by the autouse
# fixture in conftest.py.
POOL: Any = None


def _of_worker(field: "Field") -> "Worker":
    return POOL.of_worker(field.case)


def _neon_worker(field: "Field") -> "Worker":
    return POOL.neon_worker(field.case, field.executor)


def _names(fields: "tuple[Field, ...]") -> tuple[str, ...]:
    return tuple(field.name for field in fields)


class Worker:
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

    def _request(
        self,
        func: str,
        args: tuple[str, ...],
        scheme: str | None,
        data: np.ndarray | None,
    ) -> Path:
        self._count += 1
        out = self._out_dir / f"{self._count:04d}.npy"
        request = {"func": func, "args": list(args), "scheme": scheme, "out": str(out)}
        if data is not None:
            data_path = self._out_dir / f"{self._count:04d}_in.npy"
            np.save(data_path, data)
            request["data"] = str(data_path)
        assert self._proc.stdin is not None and self._proc.stdout is not None
        self._proc.stdin.write(json.dumps(request) + "\n")
        self._proc.stdin.flush()
        # the worker's stdout carries OpenFOAM/NeoN log noise; the reply line
        # is prefixed
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
        self, func: str, args: tuple[str, ...], scheme: str | None = None
    ) -> np.ndarray:
        """Run an op that produces an array result."""
        return np.load(self._request(func, args, scheme, None))

    def command(
        self, func: str, args: tuple[str, ...], data: np.ndarray | None = None
    ) -> None:
        """Run an op for its side effect, optionally shipping ``data`` along."""
        self._request(func, args, None, data)

    def close(self) -> None:
        if self._proc.poll() is None and self._proc.stdin is not None:
            try:
                self._proc.stdin.write('{"exit": true}\n')
                self._proc.stdin.flush()
                self._proc.wait(timeout=60)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                self._proc.kill()
        self._log.close()


class PybFvc:
    """``pybFoam.fvc`` — explicit operators, evaluated in the pybFoam worker."""

    def interpolate(self, field: "Field") -> np.ndarray:
        return _of_worker(field).evaluate("fvc.interpolate", (field.name,))

    def flux(self, field: "Field") -> np.ndarray:
        return _of_worker(field).evaluate("fvc.flux", (field.name,))

    def grad(self, field: "Field") -> np.ndarray:
        return _of_worker(field).evaluate("fvc.grad", (field.name,))

    def div(self, *fields: "Field", scheme: str = "linear") -> np.ndarray:
        return _of_worker(fields[0]).evaluate("fvc.div", _names(fields), scheme)

    def laplacian(self, gamma: "Field", field: "Field") -> np.ndarray:
        return _of_worker(field).evaluate("fvc.laplacian", (gamma.name, field.name))


class PybFvm:
    """``pybFoam.fvm`` — implicit operators, returned as the matrix applied to
    the current field (``M & psi``)."""

    def div(self, *fields: "Field", scheme: str = "linear") -> np.ndarray:
        return _of_worker(fields[0]).evaluate("fvm.div", _names(fields), scheme)

    def laplacian(self, gamma: "Field", field: "Field") -> np.ndarray:
        return _of_worker(field).evaluate("fvm.laplacian", (gamma.name, field.name))


class PybBackend:
    """The pybFoam (OpenFOAM) backend."""

    def __init__(self) -> None:
        self.fvc = PybFvc()
        self.fvm = PybFvm()


class NeonExp:
    """``neon.exp`` — explicit DSL operators (``nfb.evaluate_explicit``)."""

    def grad(self, field: "Field") -> np.ndarray:
        return _neon_worker(field).evaluate("exp.grad", (field.name,))

    def div(self, *fields: "Field", scheme: str = "linear") -> np.ndarray:
        return _neon_worker(fields[0]).evaluate("exp.div", _names(fields), scheme)

    def laplacian(self, gamma: "Field", field: "Field") -> np.ndarray:
        return _neon_worker(field).evaluate("exp.laplacian", (gamma.name, field.name))


class NeonImp:
    """``neon.imp`` — implicit DSL operators, assembled via
    ``nfb.evaluate_implicit`` and returned as ``(A·psi - b) / V``."""

    def div(self, *fields: "Field", scheme: str = "linear") -> np.ndarray:
        return _neon_worker(fields[0]).evaluate("imp.div", _names(fields), scheme)

    def laplacian(self, gamma: "Field", field: "Field") -> np.ndarray:
        return _neon_worker(field).evaluate("imp.laplacian", (gamma.name, field.name))


class NeonBackend:
    """The neon (NeoN) backend.

    Surface results (``interpolate``, ``flux``) keep NeoN's layout — internal
    faces first, boundary faces appended — the tests slice to the OpenFOAM
    length.
    """

    def __init__(self) -> None:
        self.exp = NeonExp()
        self.imp = NeonImp()

    def interpolate(self, field: "Field") -> np.ndarray:
        return _neon_worker(field).evaluate("interpolate", (field.name,))

    def flux(self, field: "Field") -> np.ndarray:
        return _neon_worker(field).evaluate("flux", (field.name,))


pyb = PybBackend()
nb = NeonBackend()
