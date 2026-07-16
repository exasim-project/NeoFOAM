# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""OpenFOAM discretization scheme models with validated parsing."""

from .interpolation import (
    InterpolationScheme,
    Linear,
    Upwind,
    LinearUpwind,
    LimitedLinear,
    VanLeer,
    Minmod,
    SuperBee,
    MUSCL,
    QUICK,
)
from .sn_grad import (
    SnGradScheme,
    Corrected,
    Uncorrected,
    Orthogonal,
    LimitedSnGrad,
)
from .ddt import (
    DdtScheme,
    Euler,
    Backward,
    SteadyState,
    LocalEuler,
    CrankNicolson,
)
from .grad import (
    GradScheme,
    GaussGrad,
    LeastSquaresGrad,
)
from .div import (
    DivScheme,
    NoneDiv,
    GaussDiv,
    BoundedGaussDiv,
)
from .laplacian import (
    LaplacianScheme,
    GaussLaplacian,
    NoneLaplacian,
)

__all__ = [
    # Interpolation
    "InterpolationScheme",
    "Linear",
    "Upwind",
    "LinearUpwind",
    "LimitedLinear",
    "VanLeer",
    "Minmod",
    "SuperBee",
    "MUSCL",
    "QUICK",
    # Surface-normal gradient
    "SnGradScheme",
    "Corrected",
    "Uncorrected",
    "Orthogonal",
    "LimitedSnGrad",
    # DDT
    "DdtScheme",
    "Euler",
    "Backward",
    "SteadyState",
    "LocalEuler",
    "CrankNicolson",
    # Gradient
    "GradScheme",
    "GaussGrad",
    "LeastSquaresGrad",
    # Divergence
    "DivScheme",
    "NoneDiv",
    "GaussDiv",
    "BoundedGaussDiv",
    # Laplacian
    "LaplacianScheme",
    "GaussLaplacian",
    "NoneLaplacian",
]
