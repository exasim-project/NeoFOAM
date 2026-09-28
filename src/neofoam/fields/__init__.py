# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Field declaration surface: typed BCs, value-type markers, ``Model.field(...)``.

This package implements the Phase-1 plan in
``plans/implement_field_bcs.md``: a model declares its on-disk fields
(``0/<name>``) at registration time via :meth:`ModelSpec.field`, carrying
dimensions, value type, and a discriminated union of allowed boundary
conditions. ``configurations(solver)`` surfaces the resulting per-field
schemas alongside the dictionary configs so the agent layer fills them
in the same pass.
"""

from neofoam.fields.bc import (
    AlphatWallFunctionBC,
    CalculatedBC,
    CyclicAMIBC,
    CyclicBC,
    EmptyBC,
    FixedFluxPressureBC,
    FixedValueBC,
    GenericBC,
    InletOutletBC,
    MovingWallVelocityBC,
    NoSlipBC,
    PatchBC,
    PressureInletOutletVelocityBC,
    SlipBC,
    SymmetryBC,
    SymmetryPlaneBC,
    TurbulentIntensityKineticEnergyInletBC,
    TurbulentMixingLengthDissipationRateInletBC,
    TurbulentMixingLengthFrequencyInletBC,
    WallFunctionBC,
    ZeroGradientBC,
    build_bc_union,
)
from neofoam.fields.decl import FieldDecl
from neofoam.fields.loader import load_fields, save_fields
from neofoam.fields.schema import schema_for
from neofoam.fields.value_types import (
    Scalar,
    Tensor,
    Vector,
    default_uniform,
    parse_uniform,
    zero_uniform,
)

__all__ = [
    "AlphatWallFunctionBC",
    "CalculatedBC",
    "CyclicAMIBC",
    "CyclicBC",
    "EmptyBC",
    "FieldDecl",
    "FixedFluxPressureBC",
    "FixedValueBC",
    "GenericBC",
    "InletOutletBC",
    "MovingWallVelocityBC",
    "NoSlipBC",
    "PatchBC",
    "PressureInletOutletVelocityBC",
    "Scalar",
    "SlipBC",
    "SymmetryBC",
    "SymmetryPlaneBC",
    "Tensor",
    "TurbulentIntensityKineticEnergyInletBC",
    "TurbulentMixingLengthDissipationRateInletBC",
    "TurbulentMixingLengthFrequencyInletBC",
    "Vector",
    "WallFunctionBC",
    "ZeroGradientBC",
    "build_bc_union",
    "default_uniform",
    "load_fields",
    "parse_uniform",
    "save_fields",
    "schema_for",
    "zero_uniform",
]
