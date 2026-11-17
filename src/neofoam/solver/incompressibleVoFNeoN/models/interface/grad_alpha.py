# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The interface from the phase-fraction gradient (interfaceProperties::calculateK)."""

from typing import Any

import pybFoam as pyf

from neofoam import neofoam_bindings as nfb


def _check_no_curvature_smoothing(alpha_name: str) -> None:
    """calculateK smooths alpha before the gradient when nAlphaSmoothCurvature > 0 (not ported)."""
    solvers = pyf.dictionary.read("system/fvSolution").subDict("solvers")
    if solvers.subDict(alpha_name).getOrDefault[int]("nAlphaSmoothCurvature", 0) > 0:
        raise NotImplementedError(
            "incompressibleVoFNeoN: nAlphaSmoothCurvature > 0 is not implemented"
        )


class gradAlphaInterface:
    """interFoam's interface: ``nHatf`` from the gradient of alpha1, ``K = -div(nHatf)``.

    ``nHatf = (gradAlphaf/(|gradAlphaf| + deltaN)) & Sf`` with ``gradAlphaf`` the linear
    interpolate of ``grad(alpha1)``, as ``interfaceProperties::calculateK`` without the
    contact-angle correction. The default and only member of :class:`interfaceModelNeoN`,
    which is how a case should select it. The fields are allocated once and overwritten by
    ``update``.

    Example::

        interface = gradAlphaInterface(alpha1, runtime)
        interface.update(alpha1, runtime)  # after the alpha advection of a step
        interface.curvature
    """

    def __init__(self, alpha1: Any, runtime: Any) -> None:
        _check_no_curvature_smoothing(alpha1.name)
        self.n_hat_f: Any = nfb.interface_nhatf(runtime, alpha1)
        self.curvature: Any = nfb.create_uniform_volume_field(runtime, "interfaceProperties:K", 0.0)

    def update(self, alpha1: Any, runtime: Any) -> None:
        """Recompute ``n_hat_f`` and ``curvature`` from ``alpha1``."""
        nfb.interface_nhatf(runtime, alpha1, out=self.n_hat_f)
        nfb.interface_curvature(self.n_hat_f, out=self.curvature)
