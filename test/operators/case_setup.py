# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Test-facing case access: ``simulation(mesh, executor)`` → fields as numpy.

A ``Simulation`` is one staged case bound to one executor. It exposes the
mesh (plain numpy) and loads any field present in the case's ``0/`` dir.
Fields read and write as numpy — ``np.asarray(T)`` fetches the internal
field, ``T[:] = values`` writes it back on **both** backends: the pybFoam
worker updates ``internalField`` + boundary conditions and persists
``0/<name>``; the neon worker reloads exactly those values onto its executor
(host→device copy when that executor is the GPU). Assignment is the sync
point — mutating a fetched numpy array alone changes nothing.

``flux(U)`` derives ``phi = fvc::flux(U)`` from the current ``U`` on both
backends and returns its field handle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

import backends


class Field:
    """One named field of a staged case, realized in both backend workers."""

    def __init__(self, name: str, case: str, executor: str):
        self.name = name
        self.case = case
        self.executor = executor

    def __array__(self) -> np.ndarray:
        """The internal field values, fetched from the pybFoam worker."""
        return backends.POOL.of_worker(self.case).evaluate("field.get", (self.name,))

    def __setitem__(self, key: slice, values: np.ndarray) -> None:
        """Write values into the field on both backends (device copy on GPU)."""
        updated = np.asarray(self)
        updated[key] = values
        backends.POOL.of_worker(self.case).command(
            "field.set", (self.name,), data=updated
        )
        backends.POOL.neon_worker(self.case, self.executor).command(
            "field.reload", (self.name,)
        )


@dataclass(frozen=True)
class Mesh:
    """Mesh geometry as plain numpy arrays."""

    cell_centres: np.ndarray


@dataclass(frozen=True)
class Simulation:
    """One staged case bound to one executor; loads any field in its 0/."""

    case: str
    executor: str
    mesh: Mesh

    def field(self, name: str) -> Field:
        zero = backends.POOL.case_dir(self.case) / "0" / name
        if not zero.is_file():
            raise ValueError(f"no 0/{name} in case {self.case}")
        return Field(name, self.case, self.executor)


def simulation(mesh: str, executor: str) -> Simulation:
    """The session-cached case/worker pair for ``mesh`` × ``executor``.

    Skips the test when the executor cannot run on this host.
    """
    pool = backends.POOL
    if executor == "GPU" and not pool.gpu_available():
        pytest.skip("no GPU executor available on this host")
    return Simulation(mesh, executor, Mesh(pool.cell_centres(mesh)))


def flux(u: Field) -> Field:
    """Derive ``phi = fvc::flux(U)`` from the current ``U`` on both backends."""
    backends.POOL.of_worker(u.case).command("flux.update", (u.name,))
    backends.POOL.neon_worker(u.case, u.executor).command("flux.update", (u.name,))
    return Field("phi", u.case, u.executor)
