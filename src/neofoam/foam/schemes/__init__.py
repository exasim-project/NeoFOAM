# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""OpenFOAM discretization scheme models with validated parsing."""

from .ddt import (
    Backward,
    CrankNicolson,
    DdtScheme,
    Euler,
    LocalEuler,
    SteadyState,
)
from .div import (
    BoundedGaussDiv,
    DivScheme,
    GaussDiv,
    NoneDiv,
)
from .grad import (
    CellLimitedGrad,
    GaussGrad,
    GradScheme,
    LeastSquaresGrad,
)
from .interpolation import (
    MUSCL,
    QUICK,
    InterpolationScheme,
    LimitedLinear,
    Linear,
    LinearUpwind,
    Minmod,
    SuperBee,
    Upwind,
    VanLeer,
)
from .laplacian import (
    GaussLaplacian,
    LaplacianScheme,
    NoneLaplacian,
)
from .sn_grad import (
    Corrected,
    LimitedSnGrad,
    Orthogonal,
    SnGradScheme,
    Uncorrected,
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
    "CellLimitedGrad",
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
