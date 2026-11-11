# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The NeoN finite-volume discretisation behind heatConductionNeoN."""

from typing import Any

import neon._neon as nn

from neofoam import neofoam_bindings as nfb

from .configs import ThermalPropertiesConfig


class ConductionProblem:
    """Transient heat conduction ``dT/dt = laplacian(k/(rho cp), T) + Q/(rho cp)``.

    :meth:`solve` always starts from the old time :meth:`start_step` set, so a coupling
    iteration can re-solve the step; each of ``patches`` adds its terms. One per run.

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
        # NeoN's solve, unlike OpenFOAM's, leaves a zeroGradient patch at its old values
        self.T.correct_boundary_conditions()
        for patch in self.patches:
            patch.solved()
