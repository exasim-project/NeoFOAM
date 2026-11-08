# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The NeoN finite-volume discretisation behind heatConductionNeoN."""

from typing import Any

import neon._neon as nn  # NeoN Python bindings

from neofoam import neofoam_bindings as nfb  # NeoFOAM Python bindings

from .configs import ThermalPropertiesConfig


class ConductionProblem:
    """Transient heat conduction ``dT/dt = laplacian(alpha, T) + Q/(rho cp)``.

    ``alpha = k / (rho cp)``. The ``ddt(T)`` and ``laplacian(alpha,T)`` schemes of
    ``system/fvSchemes`` (Euler, or ``steadyState`` for a steady solve). Each patch
    of ``patches`` adds its terms to the right-hand side (a coupled Neumann side its
    heat flux). :meth:`start_step` makes the current ``T`` the old time and
    :meth:`solve` computes ``T`` one step past it, so a step can be re-solved (a
    pass of the region's coupling op). One instance per run.

    Example::

        problem = ConductionProblem(runtime, T, properties)
        problem.start_step()
        problem.solve()
    """

    def __init__(self, runtime: Any, T: Any, properties: ThermalPropertiesConfig) -> None:
        self.runtime = runtime
        self.T = T
        self.properties = properties
        self.patches: list[Any] = []
        self.rho_cp = properties.rho * properties.cp
        self._alpha = nfb.create_uniform_surface_field(runtime, "alpha", properties.k / self.rho_cp)
        self._source = None
        if properties.heatSource != 0.0:
            self._source = nfb.create_uniform_volume_field(
                runtime, "heatSource", properties.heatSource / self.rho_cp
            )

    def start_step(self) -> None:
        """Make the current ``T`` the old time of the next :meth:`solve`."""
        nn.rotate_old_times(self.T)

    def solve(self) -> None:
        """Solve for ``T`` one step of the runtime's ``dt`` past the old time."""
        equation = nn.imp.ddt(self.T) - nn.imp.laplacian(self._alpha, self.T)
        if self._source is not None:
            equation = equation - nn.exp.source(self._source)
        for patch in self.patches:
            for term in patch.terms():
                equation = equation - term
        nfb.PDESolverScalar(equation, self.T, self.runtime).solve()
        # OpenFOAM's fvMatrix::solve corrects the boundary of the solved field, NeoN's
        # does not: without it a zeroGradient patch keeps the old face value. A coupled
        # patch is ``calculated``: its values stay.
        self.T.correct_boundary_conditions()
